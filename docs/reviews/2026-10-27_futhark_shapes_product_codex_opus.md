# Review: the shape kingdom's product, and the search that implements it

**Date:** 2026-10-27. Three reviews of the same work, and the disposition of
every finding.

* **Fable** reviewed the first draft of `docs/PLAN_futhark_shapes_product.md`
  and found 15 problems, of which three were blocking.
* **Codex** (`gpt-5.6-terra`) and **Opus** reviewed the implementation: the
  cost model, the GPU measurement, the loop, the generator's new rules and
  witness tests. Codex found 6 (3 blocking), Opus 22 (9 blocking).

The findings that changed the work are below. Where a finding was **not**
acted on, that is said, and why.

## Fable, on the plan

| # | finding | disposition |
|---|---|---|
| 1 | **Blocking.** The correctness half measures the wrong thing: a candidate must equal the **reference kernel's** value, not its own. | Acted on. The reference is evaluated once and every child is compared with it. |
| 2 | **Blocking.** The static cost rewards rewrites Futhark already performs: `map id` and `transpose∘transpose` are the simplifier's own rewrites, and fusion is the SOAC fusion pass's. | Acted on, and it changed the design: the cost that decides is now **measured on the GPU**, and a new rule — the transposed product — was added because it is the one rewrite the compiler cannot see on its own. |
| 3 | **Blocking.** A1–A4 have no cheaper equivalent in the vocabulary, A3 is not a legal gene, and A5 fails the correctness gate. | Acted on. The kernel set is now A1, A2, A6 and **A7** (the product computed twice), and A3/A4/A5 are gone with the reasons recorded. |
| 4 | Should-fix. A3 is not a legal gene: the kingdom has one root class. | Acted on (A3 dropped). |
| 5 | Should-fix. "The GPU backends wait on the upstream fix" is false on this machine. | Acted on. `futhark opencl` runs on this M3 Max, and the plan says so. |
| 6 | Should-fix. The "tiler-shaped" rule is vacuous and unimplemented. | Acted on, differently than first planned: the tiler's state is **measured** and reported, and the hard gate is that the program **runs** — which is stronger, and is what the measurement showed matters. |
| 7 | Should-fix. The crossover already has a home. | Acted on: `typed_gep`'s `swap_range` and `invert` are the crossover. |
| 8 | Should-fix. The composition row needs no `let` block. | Noted; the composition row is not needed — the transposed-product rule is, and it needs no new row. |
| 9 | Should-fix. B3's "bit for bit" ordering was never measured. | Acted on: Tier B is deferred, and the `exp` warning is recorded. |
| 10 | Should-fix. Tier B's `exp` breaks the bit-identical contract. | Acted on: recorded in the plan. |
| 11 | Should-fix. The static cost is not yet the extractor's cost. | Acted on: the polynomial is the cost the loop ranks with. |
| 12 | Observation. "No 2-D pathological program" rests on three compiles. | Acted on: the plan says "cut", and says why. |
| 13 | Observation. The allocation line prices a copy Futhark may not make. | Acted on: the model is a pre-optimisation model, and says so. |
| 14 | Observation. The compile cost per child is unstated. | Acted on: the measured cost is for the elites only. |
| 15 | Observation. The renderer attribution. | Noted. |

## Codex, on the implementation

| # | finding | disposition |
|---|---|---|
| 1 | **Blocking.** The measured GPU cost does not decide selection. | Acted on: the measured cost is what the elites are chosen by, and the report's best is the measured best. |
| 2 | **Blocking.** The tiling gate is not implemented. | Acted on: the tiler's state is measured (a kernel *using* `reg_tile_size`) and reported; the hard gate is that the program runs on the GPU. |
| 3 | **Blocking.** "Correct" means equality on one input map. | Acted on: the correctness is decided on a **suite** of independent inputs, and bit for bit. |
| 4 | Should-fix. `parse_impgpu` silently drops a device allocation with a non-literal size. | Acted on: such an allocation is counted, and the count of the computed ones is reported beside the literal bytes. |
| 5 | Should-fix. The fusion test constructs invalid trees. | Acted on: the trees are legal now, and the test asserts the exact traffic and allocation figures. |
| 6 | Should-fix. The −43.5 % result has no repository evidence. | Acted on: the capture is in `docs/measurements/2026-10-27_futhark_shapes_A7.md`, with the machine, the Futhark version and the command — and the number is the corrected one (15 → 9 device allocations, −40 %). |

## Opus, on the implementation

Codex's three blocking findings and Opus's first three are the same, and all
are acted on above. The rest:

| # | finding | disposition |
|---|---|---|
| 4 | **Blocking.** Device mallocs with a computed size are silently dropped. | Acted on (with Codex's 4). |
| 5 | **Blocking.** Device bytes add up mutually exclusive code versions. | Acted on: the byte total is reported as a lower bound, and the exact figure — the count of the allocation sites — is the first key of the ordering. |
| 6 | **Blocking.** The model-versus-GPU claim compares the wrong pair. | Acted on: the plan's claim is corrected — the model and the GPU *agree* about scaling an operand; the disagreement is the one the measurement shows, that the reference's kernel does not run. |
| 7 | **Blocking.** The plan's static-model numbers are wrong. | Acted on: the traffic and allocation figures are recomputed (the allocations fall to a **quarter**, not a half). |
| 8 | **Blocking.** The search does not start from a random population. | Acted on: the plan now says plainly that the reference is seeded as the incumbent. |
| 9 | **Blocking.** `let _ =` used to silence warnings. | Acted on: the two lines and their imports are gone. |
| 10 | Should-fix. The correctness comparison is loosened by a reason that does not apply. | Acted on: `identical` throughout, because both sides are the interpreter. |
| 11 | Should-fix. The backend oracle the plan describes is not built. | Acted on: the plan now says it is *not* built, and that the GPU's numbers are a report, not a gate. |
| 12 | Should-fix. The tournament ignores agreement between two wrong programs. | Acted on: the tournament uses the same order as the sort. |
| 13 | Should-fix. Polished wins in the last generation are lost. | Acted on: the best is taken after the polish. |
| 14 | Should-fix. Duplicate elites. | Acted on: the candidates and the elites are deduplicated by gene. |
| 15 | Should-fix. One GPU failure aborts the run; a fixed shared directory. | Acted on: the measurement directory is per-process, and a failed measurement is skipped, not fatal. |
| 16 | Should-fix. The cost test is vacuous. | Acted on (with Codex's 5). |
| 17 | Should-fix. The witness's "second implementation" is the same code. | Acted on: the witness's docstring now says what it really catches — a transcription slip, not a modelling error. |
| 18 | Should-fix. The new product rules have no settle test and no compiler check. | Acted on: `the_ruleset_settles` and `the_egraph_finds_the_transposed_product` are new tests, and the compiler check is `every_egraph_rule_is_an_equivalence`, which now covers all **54** rules and is green. |
| 19 | Observation. The static model's fusion rule goes further than its doc. | Acted on: the doc says it reaches any producer, because that is what Futhark's own fusion does. |
| 20 | Observation. New `_`-prefixed bindings. | Acted on: the bindings are gone. |
| 21 | Observation. Stale or banned wording. | Acted on: the cost module's header no longer says the GPU time "joins the score", and the GPU module no longer says a program was "refused" — it says what failed. |
| 22 | Observation. Correctness on one input at one size. | Acted on (with Codex's 3): a suite of inputs. |

## The two findings that changed the product

**The tiler and the allocations pull in opposite directions.** The
reference's kernel is block+register tiled and the winner's is not, so the
first draft's rule — credit a candidate only when the compiler still tiles
it — would have rejected the only win. The measurement says the tiler's
state is a *fact to report*, not a filter.

**And the reference does not run on the GPU at all.** Its OpenCL kernel is
too big for the device to compile (`createKernel: newComputePipelineState
failed`, error −48) while the winner's runs. So the hard gate is not the
cost: it is whether the program runs, and the product's story is turning a
kernel the GPU cannot compile into one it can.
