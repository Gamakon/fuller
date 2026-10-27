# Plan: the shape kingdom as a Futhark superoptimiser

**Status:** 2026-10-27, revised after a review of the first draft and after
the first run. The machinery this document builds on is written:
`src/futhark_shapes.rs` (the interface and the renderer),
`src/futhark_shapes_interp.rs` (the oracle), `src/futhark_shapes_egraph.rs`
(the generated e-graph), `src/futhark_shapes_cost.rs` (the static cost),
`src/futhark_shapes_gpu.rs` (the measured GPU cost), and
`src/futhark_shapes_evolve.rs` (the loop).

## 1. The product

**Superoptimise shape-polymorphic Futhark kernels.** Given a reference
kernel, search for a program that computes the same values with less work —
the Souper/STOKE idea, applied to Futhark, aimed at the GPU. Nothing of the
kind exists for Futhark.

The gap is concrete: **Futhark 0.28.1 has no `matmul` intrinsic at all** — not
in the prelude, not in the compiler's source (`src/` has no `matmul` and no
`MatMul`). Every matmul in Futhark is a user-written nested map/reduce.

**And the GPU backends tile that shape for you.** `gpuPipeline` runs
`tileLoops` (`src/Futhark/Optimise/TileLoops.hs`), and the block+register
tiler (`src/Futhark/Optimise/BlkRegTiling.hs`) names SGEMM as one of its
target patterns — a reduction quasi-perfectly nested in a kernel with two
parallel dimensions, which is exactly what the matmul helper is. So the
blocked matmul the C backend lacks is the GPU backend's own work, and the
search's business is the part the compiler does not do.

So the product has two halves:

* a **kernel set** — the workloads (§2);
* a **search** — GEP for discovery, the e-graph for local polish (§4).

## 2. The kernel set

This is the product's definition, so it is written down. The first draft
listed nine kernels in two tiers; a review of the draft against the code
(**`docs/reviews/`**, the disposition of every finding is in §7) showed that
five of them could not be won, and one was not even a legal gene. What
follows is the set as it now stands.

### The vocabulary, as built

Eleven types (nine matrix classes, two function classes) and seventy-one
rows:

| rows | count | what they are |
|---|---:|---|
| `map_rows@C` | 9 | map a row function over the rows of a class |
| `map2_rows@C` | 9 | map a binary row function over pairs of rows |
| `transpose@C` | 9 | swap the roles of a class |
| `matmul@mat.X.mxmat.m.Y` | 27 | the helper: `mat.n.m × mat.m.p → mat.n.p` |
| terminals | 11 | one input per matrix class, plus two `const` rows |
| row functions | 6 | unary `id`, `add1`, `scale2`; binary `id`, `add1`, `add` |

So the writable space is *matrices of rows, mapped, transposed and multiplied*.
Every kernel below is measured against that.

### The set

| # | kernel | shape | can the search beat it? |
|---|---|---|---|
| A1 | baseline matmul | `mat.n.m × mat.m.p → mat.n.p` | no cheaper equivalent in the vocabulary |
| A2 | matmul with a transposed operand | `A[n][m] · Bᵀ` | no |
| A6 | scale before or after the product | `matmul (map (*2) A) B` | only when `m < p`, and the GPU disagrees with the model here (see §3) |
| **A7** | **the product computed twice** | `map2 (add) (A·B) ((Bᵀ·Aᵀ)ᵀ)` | **yes — and this is the kernel the product is built on** |

**A7 is the one that works**, and the reason is exact. The second product is
the same matrix as the first, but the compiler cannot see that: the two are
different expressions, so its CSE leaves both and the GPU program does the
work twice. The e-graph's **transposed-product** rule — `(Y' · X')' = X · Y` —
rewrites the second operand into the first, and the compiler's own CSE then
does the rest. The rewrite is bit-identical (the same products, summed over
the same inner role in the same order), and it is checked against the
compiler like every other rule.

The first draft's A3 (the transposed product) is **not a legal gene**: the
kingdom has one root class, `mat.n.p`, and A3 is rooted at `mat.p.n`. Its
A4 (a chained matmul) and A5 (a product plus an add of zero) have no
cheaper equivalent in the vocabulary — and A5 cannot even pass the
correctness gate, because adding zero turns a `-0.0` into a `+0.0`, which
the bit-for-bit comparison is right to reject.

### What is not in the set, and why

**Blocked or tiled matmul is out, because Futhark owns it.** The GPU pipeline
tiles a nested map/reduce on its own — `tileLoops` in `gpuPipeline`, and the
block+register tiler's own header names SGEMM as one of its patterns — so
evolving tiling would be fighting the compiler and, on the GPU, losing to it.
(The *class* model would also forbid it: a block's extent is not one of the
interface's roles, so slicing and concatenation cannot be shape classes. That
is the dynamic-dims programme, `docs/ASSESSMENT_dynamic_shapes.md`. But it is
not the reason tiling is out.)

The traffic win available to the search is therefore **fusion** and
**redundancy**: fewer whole-matrix traversals, and the same work computed
once.

### The Tier B extension is deferred

Softmax and layernorm need per-row scalars, which means a 1-D class over a
role (`arr.n`, `arr.m`, `arr.p`), a reduction, a broadcast and more row
functions. That is **not** built yet, and it waits for a measured win from
A7 first: the three vector classes are added only when the kernels above have
produced one win that the compiler cannot see. Until then the class set
stays at eleven and the table does not grow.

One warning for when it is built: an `exp` row breaks the bit-identical
contract. Futhark's C backend calls libm's `expf`, and Rust's `f32::exp` is a
different implementation; only `sqrt` is correctly rounded. An `exp` row
would have to be compared on values, not bits.

## 3. The fitness: correctness, then cost

### Correctness is against the REFERENCE, not against itself

A candidate is correct when it computes the **reference kernel's** value. The
reference is evaluated once, on the inputs, and every child is compared with
that. A candidate that agrees with the interpreter of *itself* is the
kingdom's soundness test, not a fitness: every legal gene passes it, so
nothing would tie a child to the kernel it is meant to replace.

The comparison is bit for bit (`identical`) **throughout**, because both
sides are the interpreter: there is no reason to loosen it. (The *compiled*
program may differ from the interpreter where there is a reduction — Futhark's
simplifier folds `0.0 + x` away when the reduction has a single element, so
the compiled product of a `[-0.0]` row is `-0.0` where the source semantics
gives `+0.0` — but that is the compiled program's business, and it is
measured where the compiled program is run.)

And it is decided on a **suite** of independent inputs, drawn
deterministically: one sample is not enough, because a program that
*coincidentally* matches the reference on one input would enter the elite
set and be ranked.

There is also a **graded** measure — how much of a child's value agrees with
the reference's, exactly — because a random mutation is almost never equal
to the reference but is often *nearly* equal, and that is what the GEP
hill-climbs on.

### The cost is measured on the GPU, and the model is the pre-filter

**No wall-clock.** C-backend time does not transfer to the GPU: fusion and
traversal counts matter differently there, and a search that optimises for
the wrong machine is worse than no search.

Two costs, for two jobs:

1. **The static cost** (`src/futhark_shapes_cost.rs`), a polynomial in the
   dimension roles, generated into the kingdom's own table
   (`kingdoms/futhark_shapes/costs.tsv`, with the generator's own witness
   test re-deriving it). It is a *pre-optimisation* model — traffic,
   allocations, serial depth — and it is **cheap**, so it filters the
   population. It is a model and not a measurement, and it has to be read as
   one: the first draft of this document claimed it *disagreed* with the GPU
   about scaling an operand before a product, and it does not — the model
   says the scaled operand is cheaper (it touches n·m elements instead of
   n·p) and the GPU agrees (4648 device bytes against the scaled *result*'s
   4656). The real disagreement is the one §6 measures: the model cannot
   see that the reference's kernel does not run at all.
2. **The measured GPU cost** (`src/futhark_shapes_gpu.rs`), from the
   **compiled GPU program**: `futhark dev --gpu-mem --compile-imp-gpu` runs
   the whole GPU pipeline and prints the ImpGPU, and the cost is what that
   program allocates — the number of device allocation sites, the total of
   the allocation sizes that are literal, the shared memory the tiler asks
   for, and the number of kernels. This is the cost that decides, and it is
   slow, so it is for the **elites**.

   The ordering is lexicographic, and the **device allocation sites come
   first**, because they are the one figure that is always exact: the byte
   total is a lower bound (the ImpGPU holds alternative versions of the same
   work, chosen at run time by thresholds, and some allocation sizes are
   computed at run time), so it is reported beside the count of the
   unknown ones rather than in place of them.

The two disagree, and the disagreement is the lesson: the first draft's
"fusion is the win worth having" was **wrong**, because Futhark's own SOAC
fusion pass does it. Measured: `map (\xs -> xs) (transpose (transpose A))`
becomes `copy(A)` after `--simplify` alone, and `map f (map g A)` becomes
one fused map. Every rule the e-graph had before the transposed-product one
was a rewrite Futhark's simplifier already applies, so the e-graph's
candidates cost *nothing* at the compiler.

### The tiler's answer is measured, not guessed

The first draft asked whether a candidate's cost could be credited when the
tiler still fires on it — and the answer is that nothing static in the tree
can tell. The kingdom's only reduction is the fixed `MATMUL_HELPER`, and no
row can un-nest it, so every candidate that keeps the helper keeps the
tiling.

So the check is made where the question lives: the measured GPU cost
records whether the block+register tiler fired (a kernel *using*
`reg_tile_size`, which only the tiler sets — the file's header declares the
parameter whatever the program is, so matching the name alone would say
"tiled" for everything).

And the measurement says something the plan did not expect: **the tiling
and the allocations pull in opposite directions.** The reference's kernel is
block+register tiled; the winner's is not. So the tiler's state is
*reported*, beside the cost, and not used as a filter — and the hard gate
is something else, and simpler: **the program has to run on the GPU at
all.** A kernel the device cannot compile is not a program, however cheap
it looks, and the reference is exactly that: its OpenCL kernel is too big
for the device (`createKernel: newComputePipelineState failed`, error −48)
while the winner runs.

### The GPU does not reproduce the interpreter's arithmetic

Measured: the OpenCL backend's matmul of `[0.1, 0.2, 0.3]` by
`[1, 10, 100]` gives `32.100002`, where the C backend and the interpreter
both give `32.099998`. The GPU contracts the multiply-add into an FMA and
the sequential backend does not.

So the oracle is **the backend's own reference run** — and that is the next
piece of work, not something the loop does today. What the loop does today
is decide correctness on the interpreter, and the example *reports* the
GPU's values against the interpreter's, with the difference explained (the
FMA contraction). What it does not yet do is compare two GPU runs with each
other; until it does, the GPU's numbers are a report, not a gate. The
interpreter remains the oracle for the sequential C backend, where it is
exact.

## 4. The search: two operators, two jobs

**GEP mutation and crossover find new semantics.** `typed_gep` provides the
point mutation (`GeneLayout::mutate`), the homologous recombination
(`swap_range`) and the inversion (`invert`) for any kingdom, and the
kingdom's layout says what a cell may hold, so every child is legal by
construction.

**The e-graph is a local search on the elites, not the mutation operator.**
Its fifty-four rules are equivalences *by construction* — `map` with the
identity row function is the matrix, `map2` with the identity binary row
function is the first matrix, `transpose` twice is the matrix, and the
product computed the transposed way and transposed back is the product
itself — so every candidate it produces is semantically equal to its parent.
That makes it a good *cost* optimiser and a poor *discovery* engine: it
moves along the time axis only.

The loop starts from a random population **plus the reference**, which is
the incumbent it has to beat: a superoptimiser's search is over the
*equivalence class* of a program, and the reference is where it starts. The
GEP's mutations explore the neighbourhood, on the graded measure — how much
of a candidate's value agrees with the reference's — so the search can walk
towards the reference from a wrong program.

So the loop is:

1. GEP mutates and crosses over — **discovery**, on the graded measure;
2. the elite goes through the e-graph's saturation — **local polish**;
3. the candidates are written back to genes;
4. a child is kept when it is correct and cheaper.

The transposed-product rule is the one that pays, and the plan has to say
why it does not diverge the way the 1-D kingdom's fusion rule did. In 1-D, a
fusion rule together with `map id` grew the e-graph 5 → 7 → 13 → 55 → 1357
e-nodes and never converged; the fix was the unit laws of composition
(`f . id = f`, `id . g = g`) and merging the fusion rules into the core
ruleset. The transposed-product rule is *shrinking* — its right side has no
transposes — and its operands are of *other* classes, so it cannot put a
node inside the class it matches. The e-graph settles, and the measurement
is in the test suite rather than in an assumption.

## 5. The gates

1. Every kernel in the set compiles, and agrees with the interpreter —
   `identical` on the elementwise paths, `equal` on the reductions, no
   tolerance, no skip.
2. A candidate's static cost is a polynomial in the roles, computed from the
   table; the compiler's own cost is not used as the score.
3. A candidate is eligible only if the **GPU can run it**: the compiled
   program is launched on the device, not merely compiled for it. The
   tiler's state is reported beside the cost, not used as a filter, because
   the measurement shows the two pull in opposite directions.
4. A GEP generation produces heritable children: every child expresses,
   renders, compiles and agrees.
5. The e-graph's local search lowers an elite's static cost without changing
   its value, and its e-graph **settles** — a bounded saturation with a
   convergence regression test, shaped like the 1-D one.
6. Determinism: the same gene and the same seed give the same child and the
   same score.
7. The correctness is decided on a **suite** of independent inputs, and
   bit for bit, and the same is true of the polish's children (whose
   equality is proved by the rules, and checked anyway).
8. The table stays the trust boundary: `gen_tables.py --check` is byte for
   byte, and the generator's witness tests re-derive every shape relation
   from the emitted text.
9. The Tier B extension (the three vector classes) is added only after one
   measured win from the kernels above.

## 6. What the first run measured

The kernel is A7, at 64×64×64, on this machine's GPU (Apple's OpenCL over
Metal). The search starts from a random population plus the reference, and
the reference is the incumbent it has to beat.

| | device allocations | literal bytes | shared | kernels | tiled | runs on the GPU |
|---|---:|---:|---:|---:|---|---|
| the reference (`map2 (add) (A·B) ((Bᵀ·Aᵀ)ᵀ)`) | 15 | 188 416 | 15 | 10 | yes | **no** |
| **the winner** (`map2 (add) (A·B) (A·B)`) | **9** | 106 496 | 7 | 5 | no | **yes** |

**Two wins, and the second is the bigger one.** The allocations fall from 15
to 9, and the static cost from `3np + 4nm + 4mp` traffic and `np + np + nm +
mp` allocated to `np + 2nm + 2mp` and `np` — a quarter of the allocations,
not a half. The winner is the e-graph's transposed-product rule at work: the
second product became the first, and the compiler's own CSE then removed the
duplicate.

But the reference **does not run on the GPU at all**: its OpenCL kernel is
too big for the device to compile, and the winner's runs. So the search's
real win is turning a kernel the GPU cannot compile into one it can — and
the fitness has to say so, which is why "does it run" is the hard gate and
the cost is what ranks the programs that do.

The capture is in `docs/measurements/2026-10-27_futhark_shapes_A7.md`, with
the machine, the Futhark version and the command.

## 7. Cut and deferred

* **No 2-D pathological program.** Cut. The 1-D kingdom's hang is a compiler
  bug (upstream's unbounded `fuseSOACs`), not a property of two-dimensional
  shapes, and there is no 2-D instance to find. If one turns up, the fitness
  is **floored** and the search moves on.
* **The e-graph rescue stays negative.** The shape kingdom's rules cannot
  restructure the SOAC nest, and equality-preserving source rewrites were
  measured not to rescue the 1-D pathological program either. The bounded
  fusion workaround is a *compiler* workaround, and the 2-D path inherits it
  for free through the same `compile_with`.
* **The bounded-fusion workaround is C-only.** The GPU pipeline cannot be
  spelled out by hand — `futhark dev` does not expose `mergeGPUBodies` at
  all — so a GPU program that does not converge is a failed compile.
* **The population DAG** is not needed at this size: 322-word genes, and no
  shared homeotic definitions to fold. It arrives if shared matmul helpers
  across the population become the bottleneck.
* **The loop runs on the host.** The population machinery — the decode, the
  variation, the selection — belongs in phylu's GPU typed engine, and that
  is **not** built yet. What *is* built is the seam it needs:
  `futhark_shapes_evolve::score_population` scores a population of genes
  against a reference, exactly as the loop does, and
  `futhark_shapes_gpu::measure` gives a program's measured GPU cost. A
  GPU-resident population calls those two.

  The rest of the phylu work, in order:

  1. **A `SymbolTable` for the shape kingdom.** phylu's `engine.rs` has
     `SymbolTable::tsr` for its second kingdom, and it is the pattern to
     follow — but the pattern's *payload* does not fit: phylu's `Symbol`
     is a Karva operator (`Symbol::Function(op)`, `Symbol::Input`,
     `Symbol::Rnc`, `Symbol::Compound`), and the shape kingdom's rows —
     `map_rows`, `map2_rows`, `transpose`, `matmul` — are not Karva
     operators at all. So the shape kingdom needs a new operator family in
     phylu's symbol table, and the `KingdomTypes` it carries (the root
     type, the leaf types, the function types) has to carry the shape
     *classes*, not Karva's duals.
  2. **A decode stage** for it. phylu's typed device decodes a gene into a
     tree on the GPU, and the decode is type-driven, so it may work as it
     is — but it is built from the symbol table, so it has to be checked
     against the shape kingdom's classes rather than assumed.
  3. **An evaluator, or rather the absence of one.** The device's evaluator
     is the Karva/WGSL one, and it cannot evaluate Futhark programs — the
     shape kingdom's semantics are a matmul with a particular reduction
     order, and a WGSL evaluator would have to reproduce it bit for bit. So
     the shape kingdom's device is a *population without an evaluator*: the
     decode and the variation run on the GPU, and the scoring comes back to
     the host through the seam above. That is the whole reason the seam
     exists.
  4. **A fit driver** in phylu, modelled on `wgsl_typed_fit.rs`, that fills
     the device's fitness buffer from the host's scores.

  The cost of this, in one sentence: the population's *mechanics* move to
  the GPU, and its *evaluation* cannot, because the evaluation is a
  compiler.
