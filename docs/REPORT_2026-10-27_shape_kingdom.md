# Report — the shape kingdom, end to end (2026-10-27)

**What this is.** The Futhark *shape* kingdom — a kernel whose matrix shapes
are part of its type — carried from "fuller can express one" to "a population
of them evolves on the GPU, and the search finds a cheaper kernel the device
can actually run".

**Status: done and measured.** Both repos committed locally, gates green,
nothing pushed.

**Where to read it.** The GECCO **mini-poster** (one A1 page) is
`phylu/docs/paper/shape_typed_gecco.tex`; the GECCO **paper** (four pages,
ACM two-column) is `phylu/docs/paper/shape_typed_gecco_paper.tex`. Both cite
the captures in this directory, and every number in them is one of the
numbers below.

---

## 1. The commits

**fuller** — branch `feat/futhark-kingdom`, **4 commits ahead of origin, not pushed**.

| commit | what |
|---|---|
| `86429be6` | the first-class-shapes assessment (the design) |
| `9af57c9a` | the shape-classed kingdom: 11 types, 71 rows, the unification offline in `gen_tables.py` |
| `5fe8c0db` | **the search**: static cost, measured GPU cost, the transposed-product rule, the loop, the OpenCL backend, the example, the plan, the review record, the captured run |
| `f3972766` | **the seam**: `Scorer`, `fitness_of`, `score_lenient`, the public input suite |

**phylu** — branch `feat/shape-typed-kingdom` (new, off `main` at `76a2157f`), **3 commits, not pushed**.

| commit | what |
|---|---|
| `b9bf67a7` | **the typed driver**: `src/evolve/shape_typed.rs`, `examples/shape_fit.rs`, the `wgsl_typed_stage` → `walk_stage` rename, 4 new tests, `docs/tickets/0002` |
| `7f80a928` | **the GECCO mini-poster**: `docs/paper/shape_typed_gecco.tex` and its PDF |
| *the paper* | **the GECCO paper**: `docs/paper/shape_typed_gecco_paper.tex` and its PDF — the poster's content in four pages of ACM prose |

## 2. What works, measured

**The product, on the host** (64³, captured in `docs/measurements/2026-10-27_futhark_shapes_A7.md`):

| | device allocations | static traffic | runs on the GPU? |
|---|---|---|---|
| reference `map2 (add) (A·B) ((B'·A')')` | 15 | 45 056 | **no** — its OpenCL kernel is too big (error −48) |
| winner `map2 (add) (A·B) (A·B)` | **9 (−40 %)** | 20 480 | **yes** |

The e-graph's **transposed-product rule** is what wins: it rewrites the second
operand into the first, and the compiler's own CSE removes the duplicate. 27
instances, all 54 rules verified bit-identical against Futhark.

**The same win on phylu's GPU-resident population** (size 6, 16 individuals):
device draws, varies and selects; the host scores. Reference 13 device
allocations (tiled) → winner 9 (not tiled); static traffic 396 → 180; the
device's fitness 0.49 → 0.153. The device also *walks* every gene, and the
walk's node count is compared with the host's expression, every run.

**The two things the measurements refuse to fake:** the GPU's arithmetic is not
the interpreter's (the OpenCL backend contracts the multiply-add into an FMA —
758 of 4096 elements agree exactly, the rest differ by ≤3.7e-4), and the
compiler's tiler is reported, never filtered on (the reference is tiled and the
winner is not, so a "credit only a tiled form" rule would have rejected the
only win).

## 3. How the tests went

| | result |
|---|---|
| fuller full gate (`--release --features wgsl`, 55 min) | **1009 passed, 0 failed** |
| fuller clippy (`-D warnings`, `--all-targets`) | clean |
| `gen_tables.py --check` (the table is the trust boundary) | 7 files byte-identical, 12 witness tests pass |
| phylu full suite (`--release --features gpu`, 6.5 min) | **483 passed, 1 failed, 1 ignored** |
| phylu clippy (`gpu`, and `gpu,wgsl`) | clean |

**The one failure is pre-existing and not ours.** `engine::tests::three_pairs_with_the_cross_step_find_a_simple_law`
fails because the SR fit finds a model that is perfect *in-domain* and wrong
outside it. It fails identically on a **clean worktree at `76a2157f`**, with
the same model, and touches none of this work's paths. Written up as
`phylu/docs/tickets/0002` — a failure is a blocker to fix, so it is recorded
with the two things that would settle it, not waved through.

**Reviews.** fable on the plan (15 findings), codex + opus on the implementation
(6 + 22), codex (9) + fable (8) on the phylu half. Every finding is
dispositioned in `docs/reviews/`. Four of the phylu-half blockers changed the
design: the doc claimed a compile that is not on the scoring path; the correct
fitness band was not strictly ordered (so dearer-than-reference programs
tied); an empty input suite made every program "correct"; the polish
re-offered the same candidate and consumed rows it did not use.

## 4. What is *not* done, and the next steps

1. **Ticket 0002** — the pre-existing SR-engine failure above. Two questions
   settle it: is the fit's best in-domain fit the law (a search problem), or
   do the device and the host disagree about a protected operator outside its
   domain (an evaluation problem)? `examples/protected_reliance` is the
   diagnostic already built.
2. **The fuller's loose ends** — the hand-assembled pipeline against
   `futhark c`, byte for byte (the drift test); the Futhark version in the
   compile-cache key; the artefact-lifecycle fix; widening the bounded-fusion
   test's input set; fable v4's disposition.
3. **The upstream issue** — Futhark 0.28.1's `fuseSOACs` fixpoint is
   unbounded (`src/Futhark/Optimise/Fusion.hs:762`), and that is where the
   pathological 1-D program hangs. The e-graph rescue stays negative; the
   bounded-fusion workaround is a *compiler* workaround and is C-only.
4. **The H2 questions** (the wgpu host plan) — `docs/PLAN_futhark_wgpu_host.md`
   is untracked: commit the PoC or leave it? Is the C ABI permanent or a
   bridge? Vendor `rts/c` or check it out? Pin the artefact or await a release?
5. **Pushing** — never, without an explicit ask. fuller is 4 ahead, phylu 1.
