# phylu and fuller — the vision, and the split that serves it

> ## READ EVERY LINE OF THIS DOCUMENT BEFORE YOU TOUCH THE CODE.
>
> Not the section you think your task lives in. **Every line.**
>
> This is not a refactoring plan with a vision statement attached. It is a
> statement of what is being built, followed by the administrative work that
> serves it. An agent who reads §5 and starts moving files will make decisions
> that are locally sensible and globally wrong, because the constraints that
> matter are in §0 and §7, not in the file list.
>
> **Reading is not grepping** (`docs/Rules.md`). If you searched this file for a
> keyword and landed here, go back to the top.

## STATUS — where execution stands (update this block as work lands)

Execution began after the plan was verified against source and the code graph.
Sequencing call: the boundary items are resolved in fuller FIRST, while it is
one crate, so phylu's carried history includes the fixes; the carve comes after.

| step | state |
|---|---|
| Item 1 — lint tests off the engine's RNG | **DONE** `d36178c`. `lint/test_rng.rs`, bit-identical copy, cross-checked by a test. 575 + 21 green. |
| Item 2 — `Splits` in `GuardData::train` | **DISSOLVED** by the decision below — `Splits` stays in fuller with `chrom_score`. |
| Item 3 — `python.rs` bindings | **DISSOLVED** — `python.rs` stays whole in fuller; codegraph confirms it has no production use of `evolve`. |
| Item 4 — dependencies | **DONE.** phylu: `fuller` path dep, `hff`, wgpu/pollster/bytemuck (`gpu` also enables `fuller/gpu`), ratatui, serde, rayon, ndarray. No direct `egglog`. fuller drops `ratatui`, the two viewer bins, the `evolve_*` examples and `build.rs` (only the run card read it; phylu's is `PHYLU_GIT_*`). Two fuller items widened to `pub`: `extract::eval_expr_rows`, `karva::semantic_to_math`. |
| Carve phylu | **DONE.** `~/Dev/gamakon/phylu`, 156 files, 265 commits carried by filter-repo from a clone; bundle taken first. Largest blob 39 MB (historical checkpoints), under GitHub's limit. |
| Move, rewire, slim, READMEs, rename | **DONE.** `crate::{lint,gpu_eval,chrom_score,karva,snap_karva,geneframe,extract}` → `fuller::`; `hff-watch`/`hff-chart` → `phylu-sr-watch`/`phylu-sr-chart`, screenshots renamed. RNG cross-check replaced by one golden table asserted in BOTH repos. |
| Verification | **DONE.** fuller 349 + phylu 227 (226 moved + the pinned-RNG test) = 575 lib tests, + 21 viewer; zero warnings, clippy clean, both. `evolve_fit` pre-split (`22ef5d8`) vs phylu at 300 generations, seed 7014, fold+snap on: identical model, identical 661 folds / 177 snaps, identical test R² — only timings differ. |
| Push both | **DONE.** fuller `92b4a4c` to Gamakon/fuller (public). phylu to Gamakon/phylu, created **private** — the engine is the moat; flip with `gh repo edit Gamakon/phylu --visibility public`. |

**DECIDED (auto mode, reversible): `chrom_score` stays in fuller.** hff's Python
engine and notebook call `fuller.GpuSession.score_chromosomes`
(`hff/notebooks/hff_sr_engine.py:1080`,
`hff/notebooks/v1.0.4_Multidemic_SymbolicEquationRecovery.py:2179`); moving it
would break both and need a second PyO3 module. It imports nothing from the
crate and is scoring — link, wrap, least squares, metrics — which fits
"fuller: define, rewrite, evaluate". phylu calls `fuller::chrom_score`.

**The carve manifest** (§5's list was incomplete — an unlisted path is silently
left behind). phylu takes the history of: `src/evolve`, `src/bin/hff_watch.rs`,
`src/bin/hff_chart.rs`, `examples/evolve_fit.rs`, `examples/evolve_speed.rs`,
`tests/fixtures`, `experiments`, `logs`, `kingdoms`, `.claude/skills/running-fits`,
`docs/img`, `docs/paper`, `docs/audit`, `docs/EXPERIMENTS.md`,
`docs/STUDY_near_misses.md`, `docs/PLAN_engine_on_gpu.md`,
`docs/SPEC_typed_transcendental_depth.md`, this document,
`papers/HFF_Heterogeneous_Tournaments.tex`, and the crate scaffolding
(`Cargo.toml`, `Cargo.lock`, `build.rs`, `clippy.toml`, `.gitignore`, `LICENSE`,
`README.md`, `CLAUDE.md`) which is then rewritten. fuller keeps `kingdoms/`
(the symbol-table design is geneframe's) minus its `measurements/`, and the
four `logs/` scripts that drive fuller itself (`run_parity_simplify.sh`,
`run_lint_join.sh`, `apply_*.py`).

**Test count held:** 575 + 21 before and after (see Verification above).

**Use the codegraph MCP tools** (`agentic_impact`, `agentic_context`, …) for
dependency questions — Andrew's standing instruction. Caveat learned: it
resolves calls by name and cannot see `#[cfg(test)]`, so confirm a
production-vs-test back-edge in the source before acting on it.

## 0. The vision — read this before anything else

**The objective is 133 of 133.** Every ground-truth law in SRBench recovered
exactly. Not a better score, not a competitive score — all of them. Any decision
that trades a law for tidiness, speed or convenience is the wrong decision.

That objective is not aspirational padding. It is reachable *because this
project holds two instruments that no symbolic-regression system has ever had*,
and an agent who does not understand them will optimise the wrong things.

### The first instrument: an e-graph inside the evolutionary loop

Every SR system in the literature simplifies its answer **at the end**. Sympy,
by convention, as a reporting step. The reason is structural: an unsound rewrite
inside the loop corrupts the population, so nobody risks it.

`fuller` is an **egglog equality-saturation engine**. Its rewrites are *proved*
equivalent, or gated on data so behaviour cannot change. That removes the reason
nobody does this. A rewrite can therefore be a **genetic operator** — and one
gene can become an entire equivalence class of forms, all computing the same
function, differently shaped, for selection to choose between.

The design phrase is **"fuller proposes, HFF disposes."** A rewriter that
converts back into the genotype is a mechanism no competitor has, and a
simplification once found is *heritable* — crossover propagates it.

**This is not yet switched on.** §3 records that saturate-and-extract has never
been called from a generation loop. The API is built, tested and exposed; it has
no call site. That is the single largest unexploited capability in the project,
and it is the thing a new agent is most likely to mistake for a loose end.

### The second instrument: k-expressions executing directly on the GPU

Karva is a flat, level-order linearisation: a gene's child positions are
computable from the **tokens alone**, independent of the data. So threads in a
warp run the same op sequence over different rows — the property SIMD needs,
which tree-GP cannot offer.

Measured on this machine: **62.5 million gene-row evaluations per second**, and
30 million evaluations in 480 ms. One dispatch covers islands × individuals ×
genes × e-class variants × wrappers, against data uploaded once and resident.

The consequence is not "faster." It is that **exploring a wide equivalence class
becomes affordable**. Every existing bound on e-class width, beam width and
candidate count exists because scoring was expensive. It is not any more.

### Why the two together are the whole bet

An e-graph generates many forms. A GPU scores all of them at once. HFF — a
dimension-free angular fitness over many objectives — picks without a weighted
sum and without parsimony pressure. Each is interesting; together they are a
search that operates on **meaning** where GEP's own operators work on tokens.

And it generalises. A kingdom is a query over a typed symbol table, so the same
engine evolves SQL, regex or Brainfuck by changing rows, not code. **phylu** =
phylogeny + universal. That is the frontier claim, and it is why the tool is not
called `sr-engine`.

### What this means for the work in §7

The 67 unsolved laws were classified by the paper, and 53 are "the search never
reached the law." §7z lists four attacks already measured and lost: bigger
populations, differentiated swim lanes, the general mutation beam, combination
masking.

**Notice what those four have in common — every one is a conventional GP move.**
They lost because conventional GP is not what this project is. The instruments
above are absent from that list entirely.

So do not read §7z as "the 53 are unreachable." Read it as: *the ordinary
routes are closed, and the extraordinary ones have not been tried.* An agent
who concludes the remaining work is incremental has misread this document.

### The standard of work

- **Measure, do not assume.** Every number here names the run that produced it.
  A claim without a measurement is a hypothesis, and must say so.
- **A negative result is a deliverable** — recorded with its mechanism, in the
  tree. Four are in §7z, and they are why the list is honest.
- **Never delete a measurement to improve a number.**
- **Every new mechanism is a switch** (§7f), default off, in the run card. An
  A/B that cannot be run from the command line is not an experiment.
- The benchmark's verdict never steers the search. No restarts, no best-of-n.
  A 133 reached by those routes is not a 133.

---

## 1. Why the split is being done

`fuller`'s README opens: *"An e-graph engine for making symbolic expressions
smaller — provably without changing what they compute."* That is an accurate
description of about 19,000 lines. The repo is 45,000 lines. The other 26,000
are an evolutionary engine: GPU populations, island cohorts, hyperspherical
fitness selection, a telemetry stream and a full-screen terminal viewer.

A reader cannot tell what the product is. Someone arriving for the rewriter
finds a section on watching 133-law cascades; someone arriving for the engine
finds it filed under a computer-algebra library.

Both halves now work. **The best result on record is 75 of SRBench's 133**
ground-truth laws, reported in `docs/paper/hff_sr.tex` at 800 intake plus 400
champion, untyped. A cascade run during this session scored 73 at 2000+2000
with `typed_depth = 2` — a different engine (`typed_depth` did not exist at the
paper's run commit `05ef7e9`), and a lower score. Where this plan cites a
number it says which run produced it; the two are not interchangeable.

The engine has a live watcher and is about to be shown to people. It needs its
own name and its own repository before that happens.

## 2. What each product is

### `fuller` — expressions: define, rewrite, evaluate

Given an expression, produce smaller equivalent ones, provably.

- **egglog equality saturation** over bounded, non-confluent rule families
  (`algebra`, `powers`, `rational`, `sign`, `trig`, `distribute`)
- **`smallest_form`** — the canonical minimum, no data required
- **e-class variants** — *many* equivalent forms, not one. "fuller proposes,
  HFF disposes."
- **snap ⇄ concretize** — a constant as a named symbol (`pi`, `G`, `k_e`) or as
  a number, as a reversible pair
- **physics-prior mutations** — inverse-square, Lorentz, Gaussian shapes
- **the GPU linter** (`lint/`) — rule × node in bounded rounds, no e-graph;
  `SPEC_fuller_gpu.md` calls this "fuller on the GPU"
- **the typed symbol table** (`geneframe.rs`) — `Ty`, `Symbol`, `SymbolTable`,
  typed many-hot arity, `semantic_id`, kingdom-as-a-query
- **`gpu_eval`** — evaluate many expressions against resident data in one
  dispatch. fuller's `lambdify`.
- **Brainfuck** as a second target, where equivalence is decidable
- **sympy-parity scoring** — the replacement metric

### `phylu` — evolution

Given data, evolve a program that fits it. Phylogeny + universal: the aim is
any language with a symbol table, not only symbolic regression.

- **GEP populations on the GPU** — islands × individuals × genes × e-class
  variants × wrappers, one dispatch per generation
- **HFF selection** — the hyperspherical fitness function decides tournaments,
  confirmation, the stop bar (a dimension-free p-value), the beam, and whether
  a fold or a snap is kept
- **the pump** — intake and champion islands, virtual ALPS cohorts, promotion,
  arrival crossover
- **in-search editors** — fold, snap and beam as mutation operators that fire
  during evolution, not after it
- **telemetry and the watcher** — a versioned JSONL stream and
  **`phylu-sr-watch`** (renamed, §5a): verdict, islands, cohort table,
  discoveries, the hall-of-fame ladder. Kingdom-specific by design — the gene
  line and the model pane render arithmetic, so a second kingdom gets its own
  viewer rather than sharing this one.
- **run cards, checkpoints, resume** — a run is attributable and restartable

## 3. How the two relate

**Dependency flow — one direction, and this is what Cargo enforces:**

```
phylu ──depends on──> fuller
```

`phylu/Cargo.toml` lists `fuller`. fuller lists nothing of phylu's. A crate
cycle is a hard compile error, so this arrow must be clean.

**Data flow — the other way, and cyclical every generation:**

```
   phylu: a population of genes
        │  genes, as node arrays
        ▼
   fuller: the GPU LINTER — rule x node, bounded rounds, NO e-graph
        │  snapped literals, grafted forms, a guard verdict
        ▼
   phylu: score in the same dispatch, HFF picks, write back
        │
        └──────────── next generation ────────────┘
```

**Which fuller this is matters, and an earlier draft of this plan got it wrong.**
The loop calls `crate::lint::` — the linter of `SPEC_fuller_gpu.md`, matching
rule against node in bounded rounds on the device. It does **not** call the
e-graph: `grep -rn "denoise\|extract_variants\|smallest_form" src/evolve/`
returns nothing, and the paper says so in terms (`sec:rewriting`): saturate-and-
extract *"has no call site in the engine's generation loop; it was a stated
design for the search, not a component of this run."*

Production call sites into fuller, measured at HEAD (tests excluded, which an
earlier count did not do — it said 44 for `lint::node` where production is 5):
`lint::node` 5, `lint::snap_table` 2, `lint::snap_guard` 1, `lint::flat` 1.
Zero e-graph.

**And the boundary carries little traffic, which is the case for it.** The code
graph counts **207 calls crossing `evolve/` → `lint/`** against **3,672 made
inside `engine.rs` alone** — about 5%. A seam carrying 5% is a library call; one
carrying 40% would be cutting through the middle of a mechanism.

Worth recording alongside it: `engine.rs` looks like a coupling hotspot at 3,672
outgoing calls, 3.6x the next file. Normalised it is **0.40 calls per line —
identical to every other module**. It is not badly coupled, it is 9,267 lines,
four times the next largest. That is a size observation, not a design finding.

**The e-graph API is built, tested and exposed — and never wired in.**
`eclass_variants`, `denoise_candidates_assuming` and `smallest_form` all exist
in `extract.rs` with callers in examples and PyO3, but none in the generation
loop. No new code is needed to try it; a call site and a switch are
(`EVOLVE_ECLASS_EVERY`, §7f). That is the "fuller proposes, HFF disposes"
design the README claims, and it has not yet been run inside a search.

**The two flows run opposite ways, and that is the point.** phylu depends on
fuller the way a program depends on a library; fuller's *output* flows back
into phylu's population. Drawing only one arrow hides half of it.

**The interleaving is tighter than a library call.** `engine.rs:3809` passes
phylu's `GpuEvaluator` *into* fuller's snap guard, which runs fuller's rule
kernel against phylu's resident data buffer. They share a device, mid-fit.

### 3a. The return path is not what was specified — and this is the reason for §7

The design asked for, in the user's words:

> *"take the mutation and throw it in to the empty slots that we fill when we do
> the promotions … we could take all of those candidates and put them into this
> empty 80% that we cleared out. And then if they get the exact same fitness
> scores, brilliant. If they get a better fitness score, maybe we find a winner.
> If they go haywire … it just gets removed as part of regular evolutionary
> activity."*

That is a **hypothesis about diversity**: a rewritten gene is new material and
should enter as new material — into the pump's refill slots, on the pump's
beat, carrying a fresh cohort label so it is protected long enough to be
developed rather than being killed on arrival by converged elders.

What the code does instead (verified):

| specified | built | verified at |
|---|---|---|
| lands in the pump's refilled 80% | lands in the island's **worst row** | `engine.rs:4210` `fold_landing`, docstring states it |
| fires at promotion time | fires on **its own beat**, in the scoring block — `generation % fold_every == 0`. It defaults to `pump_every` so it *coincides* with the pump, but is not driven by it and does not see its slots. | `engine.rs:5616-5618` |
| gets a fresh cohort, protected to grow | **inherits its PARENT's cohort**, deliberately — "the same line, tidied … fights in its parent's ALPS band rather than wearing the label of the row it displaced" | `engine.rs:4158-4165` |
| `refill` places the discoveries | `refill` mentions folds, snaps and grafts **zero times**; it fills the 80% with fresh random rows | `engine.rs:3300` |

**The third row is a genuine design disagreement, not an oversight.** The code's
argument is that a folded gene is the same individual rewritten, so it belongs
in its parent's age band. The hypothesis says a rewritten gene is *new material*
and should enter as new material, protected by a young cohort. Both are
defensible; only a measurement settles it, and that is why 7a is an A/B rather
than a fix.

The worst-row landing was a deliberate fix to a real bug — in-place folding
lost 3 seeds out of 3, 3–40× worse on train 1−R². But it answered "where can a
fold land without destroying its host", not "how does new material enter the
population". The pump's refill slots are the answer to the second question, and
they were never wired up.

**This matters to the split** because the loop in §3 is the one thing a crate
boundary could silently break, and it is also the thing most in need of repair.
Moving the code without recording the gap would bury it.

## 4. What the reviews found

Two independent reviews (advisor, and codex against the source) agreed on a
blocking error in the first draft of this plan.

**The first draft proposed three crates** — lifting `geneframe`, `Op`, `Splits`
and the RNG into a shared `geneframe` crate — and claimed that broke the cycle.
It does not:

- `lint/snap_guard.rs:361-364` is a **production** `#[cfg(feature = "gpu")]`
  module using `gpu_eval::GpuEvaluator`, plus `:426` and `:469`. My draft only
  inventoried *ungated* back-edges and missed it.
- `lint/device.rs:19`, `snap_table.rs:680`, `snap_guard.rs:371` use
  `MAX_NODES` / `MAX_GROUPS_PER_DIM`.

So moving `gpu_eval` to phylu leaves fuller needing phylu — the cycle, recreated.

**The correction both reviews reached: `gpu_eval` stays in fuller, whole.** It
is fuller's own evaluator — `karva_to_nodes` calls `karva::karva_to_terms`, and
its parity test pins against `eval::eval_term`. Its module doc argues from GEP
population arithmetic, which is what misled the first draft; a docstring does
not set a dependency graph.

**With that, the awkwardness dissolves.** No `Splits` in a shared crate, no RNG
in a shared crate. `geneframe` stays inside fuller, because `lint/mod.rs:112`
and `:161` call `master_table()` to type-check fuller's own rules against the
Symbolic Regression kingdom — that is fuller's business, not the engine's.

**Two crates, not three.** `geneframe` has no consumer outside fuller today and
none of its other kingdoms (SQL, REGEX) are ported from nucleotable's Python.
It becomes a separate crate when it has a second consumer, not before.

## 5. The split

### Stays in `fuller` (~19,000 lines)

`extract` · `expr` · `eval` · `ruleset/` · `karva` · `snap` · `snap_karva` ·
`physics` · `score` · `parity` · `calibration` · `lint/` · `bf/` ·
**`geneframe`** · **`gpu_eval`** · **`chrom_score`** (decided in STATUS) · `python.rs` (whole)

Binaries: `parity`. Examples: the `bf_*` set, the `lint`/`parity` probes,
`measure_smallest_form`, `profile_candidates`, `snapdbg`, `00_calibration`.

### Moves to `phylu` (~26,000 lines)

| What | Lines |
|---|---|
| `src/evolve/` — engine, device, vary, write_back, telemetry, watch, chart, card, checkpoint, genealogy, hff_gpu, score, smogd, smote, umap2d | 21,701 |
| `src/bin/hff_watch.rs` → **`phylu-sr-watch`**, `src/bin/hff_chart.rs` → **`phylu-sr-chart`** (§5a) | 2,645 |
| `examples/evolve_fit.rs`, `examples/evolve_speed.rs` | — |
| `tests/fixtures/*.jsonl` — all five are telemetry recordings | — |
| `experiments/`, `logs/{cascade133.sh, report.py, tally.sh}`, `.claude/skills/running-fits`, the fit docs | — |

### 5a. The viewer is renamed, because it is not general

`hff-watch` becomes **`phylu-sr-watch`**, and `hff-chart` becomes
**`phylu-sr-chart`**.

The current name says which *fitness function* the engine uses. The new one says
which **product** owns it and which **kingdom** it serves, and the second half
is the part that matters.

**The viewer is kingdom-specific, and this is not a defect to fix.**
`gene_line_of` (`watch.rs:949`) renders `f(x_0, x_1) = …` — a mathematical
function signature, with variables in first-appearance order and literals
rounded to three significant figures. For a REGEX kingdom that line is
meaningless; for SQL it is meaningless. The model pane's bracket indenter
assumes infix arithmetic. These are SR displays, not generic infrastructure.

So when phylu gains a second kingdom, the viewer is **per kingdom**, not shared,
and the name should make that obvious rather than leaving someone to discover it
when `phylu-watch` will not render a pattern. The panels divide along a line the
rename anticipates:

| any kingdom | SR only |
|---|---|
| the verdict, the badge, the budget | the gene line at the foot |
| islands, cohorts, the trajectory sparkline | the model pane and its indenter |
| discoveries, events | the hall-of-fame ladder's HFF column |

**A consequence worth knowing before the move.** The viewer imports
`lint::node::Tree` purely to parse `raw_math`, walk it for variable names, and
round literals for display — `watch.rs:950, 973, 1791`. `Tree` lives in
`node.rs`, which imports `egglog::TermDag`, `extract::PNode` and `gpu_eval::Op`,
so **the terminal viewer transitively depends on the whole e-graph crate to
pretty-print a number.** Verified: `Tree::parse` and `Tree::to_infix` touch
neither egglog nor `PNode` — only `Op`, for operator names.

A ~120-line display parser (operator as a `String`, not an `Op`) would let
`phylu-sr-watch` build with only `serde`, `ratatui` and phylu's telemetry
schema — a binary you can hand to someone. **Not required for the split**, and
it carries a real risk of two parsers drifting; the mitigation is that the
display parser only ever renders, and `gene_line_of` already returns `None` on a
parse failure, so it can produce no answer but never a wrong one. Decide it
after the move, not during.

**Rules.md** says never rename without necessity. This clears that bar: the
binary is moving repositories regardless, so the rename costs nothing extra and
is the one moment it is free. It touches `Cargo.toml`'s `[[bin]]` stanzas, the
README, `.claude/skills/`, and every command line in the docs.

### Four things to resolve during the move

1. **`lint`'s tests use phylu's RNG.** `snap_guard.rs:824`, `snap_table.rs:991`,
   `snap_graft.rs:655` import `crate::evolve::{draw, coin, below}` for
   deterministic test data. **Fix:** a small `#[cfg(test)]` helper in fuller.
   The production RNG is phylu's exact-resume generator and stays there.

2. **`Splits` is used by a fuller public signature.** `GuardData::train`
   (`snap_guard.rs:97`) takes `chrom_score::Splits` but only reads
   `splits.total()` and `splits.n_train`. **Fix:** change the signature to take
   `train_rows: usize`; `Splits` moves to phylu with `chrom_score`.

3. **`python.rs` binds phylu types.** Lines 1550–1848 expose `GpuSession`,
   `score_chromosomes`, `gpu_predict_karva`. **Fix:** those bindings move to
   phylu's own `python.rs`; fuller keeps its lint/denoise/snap bindings.

4. **Dependencies.** phylu needs `wgpu`/`pollster`/`bytemuck` (the `gpu`
   feature is **duplicated**, not moved — `lint/` needs them too), `ratatui`
   (phylu only), `serde`, `rayon`, `ndarray`, and the `hff` path dep used by
   `engine.rs`. Check whether phylu needs `egglog` directly or can go through
   fuller's re-exports.

### phylu's git history — DECIDED: carry it

A fresh repo would drop ~150 commits of engine engineering log — the fold
operator, the pump, cohorts, virtual ALPS, the watcher, the telemetry schema.
`docs/Rules.md`: *"Commit messages are the engineering log."* Those messages
carry root causes, measurements and the reasoning behind decisions that are not
recoverable from the code.

**Decision (Andrew, this session): keep it all.**

Method: `git filter-repo` on a **clone**, never on a working repository.

```bash
git clone /Users/andrewmorgan/Dev/gamakon/fuller /tmp/phylu-carve
cd /tmp/phylu-carve
git filter-repo $(sed 's/^/--path /' manifest.txt)   # the STATUS manifest, one path per line
```

**Two hazards, both met on this machine already, both cheap to avoid:**

1. **Take a bundle first.** `git bundle create /tmp/phylu-before.bundle --all`.
   A `filter-repo` run in this session lost an entire history because
   `--invert-paths` was split onto its own line by a terminal wrap and zsh
   dropped it — without the flag the command KEEPS the listed paths and deletes
   everything else, which is the inverse of what was meant. The bundle made it
   a five-minute recovery instead of a disaster.
2. **Check the file count after, before pushing.** `git ls-files | wc -l`
   against what was expected. A filter that emptied the tree looks like success
   until you look. Never push a rewritten history without that check.

Note the two hazards point opposite ways here: this carve WANTS the listed
paths (no `--invert-paths`), where the codegraph fix wanted everything else. Say
which you mean, then verify you got it.

fuller's own history is untouched — the clone is thrown away afterwards.

## 6. Order of work

1. **phylu repo**, history carried from a clone of fuller.
2. **Resolve the four items above** in fuller first, while it is still one
   crate — so each is a small, testable commit against a green tree.
3. **Move the files**, add `fuller = { path = "../fuller" }`, rewrite imports
   (`crate::lint::` → `fuller::lint::`, and so on).
4. **Slim fuller**: drop `ratatui`, remove the moved modules from `lib.rs`.
5. **READMEs**: fuller loses the `hff-watch` section and the SR framing; phylu
   gains it plus the two screenshots in `docs/img/`.
6. **Push both.**

Path dependencies while the boundary settles; pin to a git revision once it has.

## 7. The road to 133 of 133

**The objective is every law found, not a better score.** That changes what
belongs in this section: not a defect list, but what each unsolved law NEEDS.

The paper classified all 67 misses of the cascade run (`sec:failure`, Table 4).
This is the only honest starting point, and it says the work divides unevenly:

| class | what happened | laws | touched by |
|---|---|---|---|
| **D. Ran out of time** | the search never reached the law | **53** | the search itself |
| **B. No model scored** | our string too long for the scorer to parse | 16 | the reporting path |
| **C. Hit the generation cap** | the cap ended it, not the search | 11 | one config value |
| **A. False stop** | met the bar with the wrong function | 3 | the stop bar |

Classes overlap; a law can be capped and unscored. The paper's own conclusion:

> *"Of the 67 laws unsolved before this change, **53 were the search not reaching
> the law at all, and no amount of work on how we write the answer touches
> those**. The recoverable remainder is concentrated in how the answer is written
> and when the search is allowed to stop."*

**So this section has two halves, and the first draft only had one.** Everything
7a–7g addressed is the recoverable remainder — worth perhaps 30 laws and cheap.
The 53 need the search to reach laws it has never reached, and the honest
position is that we do not know how. §7z states what has already been tried and
failed there, because at a 100% objective the negative results are the map.

**None of this blocks the split.** It is recorded here because the split is the
moment it would otherwise be lost.

### 7a. The return path — the highest-value item

As set out in §3a. The fix has three parts, and they should land together
because separately they do not test the hypothesis:

1. **Stage the candidates instead of landing them.** Today a fold overwrites a
   live row the moment it is made. It must instead be materialised into a
   staging buffer, so the pump has something to place. This is the part that
   makes the other two possible, and it is the real work.
2. **Land in the refilled 80%**, not the worst row. `refill` takes the staged
   list and places candidates ahead of the fresh random draws. A slot with no
   candidate falls back to a fresh random individual, as now; a candidate with
   no slot is **discarded** — it does not fall back to a live row, or it has
   not entered through the refill path at all.
3. **Stamp the pump beat's cohort.** `refill` already gives fresh rows
   `fitness = NaN`, no `Scored`, and the current beat's label
   (`engine.rs:3327`) — exactly the semantics wanted. But the existing
   `arrivals` path deliberately *preserves* a source cohort, so a staged fold
   cannot reuse it: it needs its own path that stamps `marks.generation`,
   invalidates the score state, and records a genealogy origin.

**Two design questions the plan does not yet answer**, both raised by review and
both needing a decision before code:

- **Routing.** Folds scan intake *and* champion islands. A fold born in a
  champion island has no refill slots of its own — the champion is not refilled.
  Which paired intake receives it, or is a champion fold left to land as it does
  today?
- **Ordering.** Folds currently run before the pump in the same generation
  (`engine.rs:5616` then `:5648`). That ordering works *if* the fold stages
  rather than lands, which is item 1.

**The measurement:** A/B against today's worst-row landing, on the laws the
cascade misses, 3 seeds. Report exact recoveries and generations-to-exact. The
hypothesis is that (3) is what makes (2) worth doing — placement without
protection is the same trap the refill already has.

### 7b. The 180-second stage buys nothing

From the completed 133 cascade (`logs/report.py`):

| stage | found | attempted |
|---|---|---|
| 60 s | 68 | 133 |
| 120 s | 5 | 65 |
| **180 s** | **0** | **60** |

The 180 s stage consumed roughly 40% of the 5.39 hours and recovered nothing.
**This is a measurement, not a property of the code** — it comes from one
cascade (seed 7014, commit `b21ea30`) read back by `logs/report.py`. It should
be reproduced on a second seed before the budget ladder is changed on it.

**The diagnosis is replay, not "time spent".** `logs/cascade133.sh` starts every
stage from generation 0 with the same `EVOLVE_SEED=7014` and no checkpoint. The
engine is deterministic, so the 120 s stage re-walks the path the 60 s stage
already walked and the 180 s stage re-walks it again: of 360 s spent on a law
that reaches the third stage, only 180 s is new search. Roughly half the
cascade's wall clock is recomputation.

**The remedy is resume, not a new seed.** A seed per stage is three independent
searches, which is what `sec:integrity` rejects as *"cheating-like"* (it scored
54). Resuming each stage from the previous stage's checkpoint keeps it one fit
per problem and turns every second of the ladder into new generations.
Checkpoint and resume already exist in the engine; the script does not use
them.

### 7c. Typed refusal does not remove shapes

`kingdoms/tsr/README.md`: at `n_genes = 3` with `gene_subsets` on, a gene
refused by the TSR depth ceiling does **not** kill its row. It rides along
unscored, still mutating and crossing into its neighbours, so typing
accumulates dead weight instead of removing shapes from the population. That is
why a typed run's depth histogram still shows entries past the ceiling.
Confirmed at `engine.rs:3072` with a regression test at `:6395`.

**It is only true with subsets on.** With `gene_subsets` off every combination
uses every gene, so a refused gene does score the row `PI` and it loses
selection pressure, as the TSR spec originally assumed (`engine.rs:6399`). The
deficiency is the interaction of the two settings, not the ceiling alone.

### 7d. The scorer's wrapper sits outside the ceiling

`WRAPPERS = [Identity, LogAbs, SqrtAbs]` is applied to the linked genes'
predictions, outside any gene, so `t_depth` never sees it. A depth-2 gene under
`SqrtAbs` is a depth-3 reported model, and the search *selects* on the wrapped
form. `Abs` is itself depth-raising, so the wrap adds two.

Two fixes, neither done:
- count the wrapper in `Scored::t_depth` — the honest instrument, and the
  precondition for the other. **Not free, as I first wrote:** with
  `Config::tower` enabled `t_depth` feeds the HFF tower penalty
  (`engine.rs:1422`, `:3175`), so a corrected depth changes selection. It is
  behaviour-neutral only with the tower off.
- admit a wrapper only if it fits the depth budget — makes the shape
  unrepresentable, but that wrap is what found `feynman_I_6_2a` at generation 6

### 7e. The snap lattice carries constants no law uses

`e`, `phi`, `sqrt2`, `sqrt3` appear in **zero** of the 133 true laws yet consume
12–20% of snap matches; roughly 42% of gene constants are snap's own write-back
being folded back in. The lattice is a physics table being used as a general
numeric one. **Also a measurement claim** — the lattice provably contains those
constants, but the match percentages come from an earlier tally that should be
cited or re-run before action.

### 7f. Every new mechanism is a switch, and the switch is in the run card

**The rule, from the existing code, not invented here.** Every in-search editor
already follows one pattern: a `Config` field, an `EVOLVE_*` environment
variable, `0` or `None` meaning off, and the value serialised into `card.json`
so a result names the configuration that produced it.

| field | env | off |
|---|---|---|
| `fold_every` | `EVOLVE_FOLD_EVERY` | `0` |
| `snap_every` | `EVOLVE_SNAP_EVERY` | `0` |
| `beam_every` | `EVOLVE_BEAM_EVERY` | `0` |
| `gene_subsets` | `EVOLVE_GENE_SUBSETS` | `false` |
| `typed_depth` | `EVOLVE_TYPED_DEPTH` | unset = untyped |

**Nothing in §7 may land without one.** An A/B is only possible if both arms are
reachable from the command line, and the paper's central claim is currently
untestable for exactly this reason — `sec:together` reports *"what the record
does not contain is a single fit with editing switched off"*, which is a missing
switch as much as a missing run.

The switches §7 requires:

- **7a** — `EVOLVE_FOLD_LANDING=worst|refill`. Not a boolean: the two landings
  are the two arms of the measurement, and the default must stay `worst` until
  the A/B says otherwise.
- **7a** — `EVOLVE_FOLD_COHORT=parent|fresh`, separately, because §3a's
  disagreement is about the cohort and not the slot. Crossing the two gives the
  2×2 the paper says is missing a row.
- **The e-graph in the loop** (§3, the diagram's error) —
  `EVOLVE_ECLASS_EVERY` with `0` as the default, since it has never run in a
  generation loop and must not switch itself on.
- **7d** — `EVOLVE_WRAPPER_IN_DEPTH=0|1`. Default `0` preserves today's
  behaviour; `1` counts the wrapper, which changes selection whenever
  `Config::tower` is on.
- **7b** — the cascade's stage ladder belongs in the runbook, not baked into
  `logs/cascade133.sh`, so the 60/120/180 split can be changed without editing
  a script mid-run.

**And the card must carry them.** `experiments/README.md` is the runbook; a
switch that exists but is not documented there is one nobody will find. The
existing table in that file is the place, and the rule stated at its head — *"if
you had to grep the source to find a setting, that is a bug in this file"* —
is the standard.

### 7g. What the paper records and this plan had missed

A review of `docs/paper/hff_sr.tex` against this plan found four open items the
paper states and §7 did not carry. They are listed here in the paper's own
terms, because each is already measured.

**38 laws we may have already solved and then broke on the way out.** This was
first written here as "a perfect 38-law predictor, unbuilt" — which framed a
DIAGNOSIS as a FILTER. Building a gate that refuses to submit those 38 recovers
**zero laws**. It converts a wrong answer into no answer. The framing was wrong
and the correction matters, because it changes the work from screening to
repair.

What `sec:failure` actually reports: of the 133, 38 produce a non-finite value
on at least one row in the **submitted** string, and not one of the 38 has ever
been solved. But the cause is ours. The engine scores the **faithful** form,
where a protected operator returns a guard value; we submit the **plain** form,
where the same expression is raw and returns NaN. `feynman_II_11_28` scores
1−R² = 8e-6 on the form the engine selected while **the string handed to the
benchmark is undefined on 59% of the rows.** Four laws are undefined on 100%.

**The specific mechanism, traced in source.** `resolve_protected`
(`engine.rs:1687-2189`) already converts protected operators to raw ones — that
is the paper's §7 and it is worth ten laws. Every conversion is justified on
`fit_rows` = train + validation (`examples/evolve_fit.rs:516-518`). But the
model is judged on **test** rows. The `Abs`-shed at `engine.rs:2043-2051` fires
when its argument is non-negative on every *fit* row; on a test row where the
sign flips, `log(-x)` is NaN, and the protected operator that would have caught
it has been removed. **The rewrite is justified on a strictly smaller row set
than the one it is scored on.**

Two corrections to the paper's own account, found in the code: `eval.rs:192`
guards `ProtectedLog` only at `x == 0` and non-finite, not on negatives; and
`node.rs:156` renders the plain form as `log(Abs(x))`, which is finite on
negatives. So the plain-NaN sites are **narrower** than the paper implies, and
the `Abs`-shed carries more of the 38 than the protected/plain split does.

**Three routes, in order of honesty:**

1. **Score what we submit.** The engine optimises one function and reports
   another; nothing anywhere evaluates the plain string for finiteness
   (`engine.rs:2550-2658` checks only the protected reference). If the plain
   form were scored alongside the faithful one, a chromosome undefined on 59% of
   rows would lose its fitness and the search would leave it on its own. This is
   the root cause; the other two are repairs.
2. **Justify a rewrite on the rows it is judged on**, or refuse it. The shed
   needs evidence covering test rows, not just fit rows.
3. **Rewrite so the guard cannot fire** — `log(Abs(x))` or `log(x²)/2` is total
   on the reals and needs no guard. That is fuller's job, and the e-graph can do
   it.

**There is a per-law discriminator**, and it is a scan over strings already on
disk: if a `Piecewise` survives in `MODEL_INFIX`, the fit genuinely depends on
guard values and the plain string is a different function — really unsolved. If
no `Piecewise` survives and the plain string is still non-finite on test rows,
it is the `Abs`-shed, and the law is recoverable. **That scan splits the 38 into
"unsolved" and "broken in reporting" and nobody has run it.**

**16 laws lost to bloat — and that is fuller's own job.** `sec:failure` Table 4,
class B: SRBench's scorer returned `None` because our string was too long to
simplify. 16 laws, *"the largest recoverable class, and it is entirely ours to
fix"*. Median 323 characters against 122 for a scoring model. Producing a
shorter equivalent form is exactly `smallest_form`.

**Four of the 75 are sympy beating fuller.** `sec:results`: *"The run scores 75
on the tidied string against 71 on fuller's own. The tidy is worth four laws …
four of the 75 are reporting wins rather than search wins."* fuller exists to
replace sympy; a four-law deficit on that task is a fuller-product deficiency.
Parity overall is 33.6%, and 13% on `simplify`.

**The two halves of the stop bar contradict each other.** `sec:hff`: the
synthetic third block *"is not meant to decide that a fit is exact — the 1−R²
condition honours that and skips it, but the angle does not, so the block's
error enters p like any other objective. **One of the two is wrong.**"* An open
defect in phylu's selection core, stated by the paper and never carried forward.

**The experiment the paper says is missing is cheaper than 7a's.**
`sec:together`: *"What the record does not contain is a single fit with editing
switched off, in either cohort condition. The two-by-two has one row, not one
cell … The experiment is cheap and we have not run it."* Both of 7a's arms have
editing ON, so 7a fills a different cell of the same row. The missing row is one
environment variable — `EVOLVE_SNAP_EVERY=0` — and it is what the paper's
central claim rests on. **It should run before 7a.**

### 7h. Smaller, recorded

- **Open audit findings 8, 10–13, G4, G7, G8** in `docs/audit/PAPER_CODE_ALIGNMENT.md`
- **Per-cohort keeping** in the pump's refill — specified, never built
- **The slot-search / Occam operator** — specified, never built
- **`logs/tally.sh` and `logs/report.py` overlap** — one live view, one record;
  they read the same directory by different means and should share a reader

### 7z. The 53 — what has been tried, and what this run adds

**This is the hard half and it has no plan yet.** Recording it honestly is worth
more than a proposal, because four routes have already been measured and lost.
Each is a road not to retake without a reason.

**Bigger populations lose.** `sec:negative`: 20,000 individuals at a 35-
generation cap scored **42 of 133 against 46** for the ordinary configuration.
The cause is measured: 58% of a 15.6 s fit is data evaluation, not population
handling, so throughput per individual does not convert into search. (That arm
also carried a redundancy objective, so it is not a clean population-only test.)

**Differentiated swim lanes lose.** Three islands under different variation
rates: the uniform arm solved 4 of 10, the differentiated arm 3 — and lost
`I_15_10`, a Lorentz law the uniform arm found at generation 186. *"A law lost
outright is a worse signal than a one-problem count difference is a good one."*

**The general mutation beam is exhausted.** Over six near misses a beam mutant
beat its original **0.05%** of the time. Changing population size alone moved
log₁₀ p by up to 1.8 decades on the same problems — more than the beam ever did.
The tree half is off by default.

**Combination masking converted nothing** (`sec:negative`).

**And the integrity rules close the cheap routes.** `sec:integrity`: no
restarts, one fit per problem; splitting a budget into three independent
searches scored 54 and was *"rejected as cheating-like"*; the benchmark's
verdict never steers the search. A 133 reached by any of those is not a 133.

#### Two things this session's cascade adds that the paper does not have

1. **`feynman_test_5` was recovered.** The paper reports the 20-problem bonus
   set as *"0 of 20 and always have"*. Our cascade found it at 60 s:
   `1.0048·sqrt(4·9.775·x_0³/(x_1·(x_2+x_3)))`, test R² 1.000000 — a real
   recovery, not a fitted blob. **1 of 20 on a set that had never scored.** The
   configuration differed (typed depth 2, 2000+2000), and one event is not a
   rate, but it is the first crack in that set and it should be reproduced on a
   second seed before anything is concluded from it.

2. **The 133 has never been run at the paper's own settings.** The paper's 75
   was 800+400 untyped; our 73 was 2000+2000 typed. Neither is a controlled
   comparison of the other, so *"typing costs two laws"* is not established —
   it is two different runs. A like-for-like pair is cheap and has not been run.

#### What is left untested, from the paper's own limitations

- **The noisy tracks** — never attempted.
- **The `1/sqrt(1 - v²/c²)` family, nine problems** — never solved in any run.
  `I_10_7` fell once to a grafted wrap at generation 7 with test R² 1.000000 on
  one seed and not another. One event, not a rate — but it is evidence the
  family is reachable, and the wrap that found it is the only mechanism that
  ever has.

**The honest statement for a 100% objective:** the recoverable ~30 are
engineering and are covered above. The 53 are a research problem, the four
obvious attacks have been measured and lost, and the two live threads are the
grafted wrap that took `I_10_7` and whatever took `test_5`. Both are single
events that need reproducing before they are anything.

## 8. Verification

Each crate, independently:

```bash
RUSTFLAGS="-D warnings" cargo test --features gpu
cargo clippy --all-targets --features gpu
```

**Baseline to hold: 574 lib tests + 21 viewer tests, zero warnings.** No test
may be dropped to make the split compile — a test that cannot move is a
boundary that is wrong.

End to end, proving behaviour is unchanged:

```bash
# phylu: the viewer reads a recorded stream (fixtures are in-repo)
./target/release/hff-watch --dump tests/fixtures/telemetry_v1.jsonl

# phylu: a fit still runs, and the fold/snap loop into fuller still fires
cargo run --release --features gpu --example evolve_fit -- <dataset>

# fuller: parity unchanged against the frozen sympy corpora
cargo run --release --bin parity -- parity/corpus/*.jsonl
```

The `evolve_fit` run is the one that matters: it exercises the data-flow loop in
§3, which is the thing a crate boundary could silently break.
