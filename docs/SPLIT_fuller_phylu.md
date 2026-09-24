# Split `fuller` into two products: `fuller` and `phylu`

## 1. Why this is being done

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
- **telemetry and the watcher** — a versioned JSONL stream and `hff-watch`:
  verdict, islands, cohort table, discoveries, the hall-of-fame ladder
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

Per-generation call counts into fuller, measured at HEAD: `lint::node` 44,
`lint::snap_table` 6, `lint::snap_guard` 5, `lint::tables` 2,
`lint::snap_graft` 2, `snap_karva::constant_values` 3. Zero e-graph.

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
**`geneframe`** · **`gpu_eval`** · `python.rs` (its own bindings)

Binaries: `parity`. Examples: the `bf_*` set, the `lint`/`parity` probes,
`measure_smallest_form`, `profile_candidates`, `snapdbg`, `00_calibration`.

### Moves to `phylu` (~26,000 lines)

| What | Lines |
|---|---|
| `src/evolve/` — engine, device, vary, write_back, telemetry, watch, chart, card, checkpoint, genealogy, hff_gpu, score, smogd, smote, umap2d | 21,701 |
| `src/chrom_score.rs` — link, wrap, LSM, metrics, the behavioural signature | 970 |
| `src/bin/hff_watch.rs`, `src/bin/hff_chart.rs` | 2,645 |
| `examples/evolve_fit.rs`, `examples/evolve_speed.rs` | — |
| `tests/fixtures/*.jsonl` — all five are telemetry recordings | — |
| `experiments/`, `logs/{cascade133.sh, report.py, tally.sh}`, `.claude/skills/running-fits`, the fit docs | — |

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

### phylu's git history

A fresh repo drops ~150 commits of engine engineering log — the fold operator,
the pump, cohorts, ALPS, the watcher. Rules.md: *"commit messages are the
engineering log."* **Recommendation:** `git filter-repo` on a *clone* to carry
the engine's history into phylu, leaving fuller's history untouched. If that
proves awkward, start fresh and say so in phylu's first commit rather than
letting it happen silently.

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

## 7. Known deficiencies, carried into phylu

These are open on the engine today. They are recorded here because the split is
the moment they would otherwise be lost, and because several are the difference
between 73/133 and a better number. **None is a blocker for the split.** Each
needs its own measurement, in the order a scientist would take them.

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
Either the budget ladder should stop at 120 s, or the third stage should change
something other than time — a different seed, a wider population, a different
kingdom. Time alone is spent.

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

**A perfect 38-law predictor, unbuilt.** `sec:failure`: of the 133, 38 produce a
non-finite value on at least one row, and *"every single solve is on the finite
side. Not one of the 38 was ever solved, in any run. In 133 laws it is a perfect
predictor with no exceptions."* The analogous ROUNDING gate is built and wired
(`lint/node.rs:178` `dies_on_rounding`, used in the final-form sort at
`engine.rs:2658`); the FINITENESS gate is not — no pre-submit finite-on-every-
row check exists anywhere in `src/evolve/` or `src/lint/`. The paper calls it
*"knowable before submitting"*. This is the cheapest item in §7 and the best
evidenced.

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
