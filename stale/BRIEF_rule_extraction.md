# Brief: Bootstrap a sympy-free symbolic-rewrite ruleset (Rust + egglog + HFF)

> Status: queued. **Do not start until Phase 1.2–1.4 of `BRIEF.md` (the
> denoise core: karva↔terms converter + real-domain evaluator) exist** — this
> work reuses them. This document supersedes the earlier
> "SymPy → egglog rule extraction" brief, which failed because it targeted
> egglog-python, used SymPy as its own oracle, and treated heuristic SymPy
> source as mechanically transcribable. None of those hold here.

## Role

You grow a small, **provably sound** egglog ruleset for symbolic rewriting,
inside the existing Rust `fuller` crate. You transcribe known mathematical
identities; you do not invent. A rule ships only if it (a) is numerically
sound on sampled real points, (b) improves the HFF-TrueNorth fitness of the
ruleset against the corpus, and (c) is **reviewed and approved by the
governing engineer** (see Governance). Coverage is not a race; correctness is
non-negotiable.

## The clean break from SymPy (read this first)

This project exists to *leave SymPy behind*. Therefore:

- **No `sympy` import anywhere** — not in rules, not in the corpus generator,
  not in scoring, not in tests. If you `import sympy`, you have failed the
  brief.
- **No SymPy as an equality oracle.** Equality of `extracted` vs `target` is
  decided two ways, neither of which is `sympy.simplify`:
  1. **Numerical soundness** — evaluate both expressions on N random real
     sample points using the crate's real-domain evaluator (`src/eval.rs`,
     built in Phase 1.3). Equal iff `max relative error < tol` across all
     points where both are finite (NaN handling: a point where one is NaN and
     the other finite is a mismatch).
  2. **Structural reach** — after saturating the ruleset on `input`, does
     `target` land in the **same e-class**? (egglog proves the equality.)

## Target & reuse

- Build inside the Rust `fuller` crate. Reuse, do not reimplement:
  - the Phase 1.0 calibration substrate (`src/calibration.rs`) — proves the
    egglog 2.0 loop works; do **not** re-run a Python calibration.
  - the karva↔terms converter and real-domain evaluator from Phase 1.2–1.4.
- egglog is the Rust crate `=2.0.0`. Extraction uses `extract_variants(.., n)`
  and a `CostModel` (see `reports/environment.md`). **Never** egglog-python.

## Modules, in order (do not interleave)

1. **algebra** — canonical algebraic identities. Smallest surface; reuses the
   denoise five (mul/add identity, mul-zero, double-neg, same-op collapse)
   plus a curated handful. **Start here.**
2. **trig** — trigonometric identities (the sound, finite, textbook set:
   Pythagorean, double-angle, even/odd, etc.).
3. **(optional, later) sympy-mining enrichment** — only if algebra+trig land
   cleanly. Mining heuristic SymPy source for *additional* candidate
   identities is research, not transcription; it is explicitly a stretch goal,
   not a deliverable. If reached, every mined candidate goes through the exact
   same soundness + governance gate as a curated one.

## Per-module workflow

### Step 1 — Corpus (no SymPy)

Generate `corpus/<module>.jsonl`: ≥200 `(input, target)` pairs, encoded in the
crate's own term representation (the converter's serialisation — **not**
srepr). Generation:

- Seed the PRNG (reproducible).
- Random expression trees, depth ≤ 4, over the module's operator/leaf set
  (leaves: a few symbols, small integers, and module constants).
- `target` = the canonical/simplest known form, obtained from the **curated
  identity definitions** (we know the target because we wrote the identity),
  not from a black-box simplifier.
- Skip pairs where `input` already equals `target` (nothing to learn).

### Step 2 — Candidate identities

The governing engineer supplies the **seed identity list** for the module
(math notation, with preconditions and direction). The agent may *propose
additional* candidates from cited references (e.g. a standard identity table),
each with: the identity in math notation, preconditions (positive base,
real arg, ≠0, …), and direction (LHS→RHS or bidirectional). Proposed
candidates are not assumed correct — they earn their place via the gate below
**and** governance review.

### Step 3 — Scoring: HFF-TrueNorth, multi-objective

Score each candidate ruleset as an objective **cost vector**, every coordinate
pre-scaled into [0,1], then projected through **HFF-TrueNorth** (the Rust HFF
library, `hff` crate / PyO3). **TrueNorth only — never HFF-Balanced.**

```
vec = [ 1 - soundness,      # soundness ∈ [0,1] : fraction of corpus numerically sound (near-veto)
        1 - reach,          # reach     ∈ [0,1] : fraction where target is in input's e-class
        sat_cost / 10_000 ] # nodes used / budget cap  (lower is better)
# Optional 4th objective (OFF by default — see Parsimony):
#       parsimony = rules_used / MAX_RULES
fitness = hff_truenorth(vec, normalize = False)   # lower fitness = better
```

**Normalization rule (load-bearing — do not deviate):** every objective is
pre-scaled to [0,1] by the agent in the documented way above, then HFF is
called with `normalize = False`. Do **not** use HFF's population min-max
(`normalize = True`) on this vector — the bounded fraction columns would
collapse onto the pole. If a new objective is unbounded, pre-scale it and
document the scaling; never reach for `normalize = True`. **If you are unsure
about normalization, STOP and ask the governing engineer** — this is the
single most error-prone part of using HFF.

**Sense:** all objectives are expressed as **costs (lower = better)** to match
TrueNorth minimizing angular distance to the pole. Maximize-objectives
(soundness, reach) enter as `1 - x`.

**Parsimony is optional and OFF by default.** Validation-based evolution
already controls overfit well, so a structural size penalty is usually
redundant pressure. Enable the 4th objective only for a run where ruleset size
specifically matters, and say so in the report.

### Step 4 — Accept / reject (one candidate at a time, simplest first)

A candidate is **accepted** iff ALL hold:
1. **Sound:** it introduces no soundness regression — no previously-sound
   corpus input becomes unsound, and the rule itself is sound on every corpus
   input where it fires. (Soundness acts as a near-veto: any unsoundness =
   automatic reject, regardless of fitness.)
2. **Net-positive:** ruleset HFF-TrueNorth fitness strictly improves.
3. **Budget-clean:** every corpus input still saturates within **1 s wall /
   10,000 nodes**.
4. **Governance-approved** (see below).

Reject otherwise; log rule, fitness delta, regressions, budget violations.

### Step 5 — Stop

- All curated (and any approved proposed) candidates exhausted, OR
- 10 consecutive rejections, OR
- governing engineer calls it.

There is **no fixed match-rate target.** A small, fully-sound ruleset that
covers less corpus is a better outcome than a large one with any unsound rule.

## Governance & quality control (the governing engineer owns this)

The governing engineer (Claude, in the parent session) is responsible for
correctness and direction. The agent **cannot ship rules unreviewed.**

- The engineer **supplies the seed identity list** per module.
- Every **accepted** rule is **reviewed by the engineer** before it lands in
  the module file: identity correct? precondition complete? direction right?
  guard sound? The agent surfaces accepted rules for review in batches; the
  engineer approves, amends, or vetoes.
- The agent **must stop and escalate** (not push through) if: calibration /
  the reused converter/evaluator behaves unexpectedly; a rule passes the
  corpus but is suspected unsound off-corpus; or normalization for HFF is
  unclear. Report the issue; do not invent a workaround.
- The agent runs **autonomously between review points** — it does not need
  prose check-ins for routine accept/reject; it batches accepted rules for
  governance and otherwise proceeds.

## Hard constraints (egglog hygiene — kept from prior art)

- **Saturation budget: 1 s wall, 10,000 nodes per input. Non-negotiable.**
  Violating rules are rejected without further analysis.
- **No bare commutativity rules** (`Add(a,b)→Add(b,a)`) and **no bare
  associativity rules.** egglog canonicalises these via e-class merging;
  encoding them as rewrites blows up saturation. (This was the trap that the
  Phase 1.0 calibration surfaced — egglog does NOT know operator algebra
  unless a rule says so, but the *fix* is a proper commutative declaration /
  canonicalisation, never a raw rewrite.)
- **RHS pattern variables must all appear on the LHS.** Reject pre-test.
- **Conditional rewrites require explicit guards** (e.g. `x/x → 1` needs a
  `≠ 0` guard). If the guard cannot be expressed in egglog 2.0, reject and log.
- **One ruleset per module file.** No cross-module rule imports; each module
  ships standalone.
- **Real domain only.** No complex-domain identities, no `re()/im()`-style
  ops. (This is the whole point of leaving SymPy: there is no complex domain
  to escape from.)

## What you do not do

- Do not import `sympy`. Anywhere.
- Do not target egglog-python. This is the Rust crate.
- Do not port `simplify()`-style strategy code — only atomic identities.
- Do not port calculus (integration, limits, series). Algebra/trig only.
- Do not add identities not justified by the curated list or a cited
  reference. Transcription, not extension.
- Do not ship a rule the governing engineer has not approved.

## Deliverables per module

1. `src/ruleset/<module>.rs` — the egglog ruleset (data-first, so the future
   rule registry can load it; one module per file).
2. `corpus/<module>.jsonl` — ≥200 `(input, target)` pairs, crate-encoded,
   seeded, depth ≤ 4, no trivial (input≡target) pairs.
3. `reports/<module>.md` — environment (sympy: ABSENT; egglog, HFF versions);
   corpus size + seed; per-rule accept/reject log with fitness deltas; final
   HFF-TrueNorth fitness; soundness summary; saturation P95 nodes; the
   governance review record (who approved what).

The deliverable is the artefacts + report, not prose progress updates.
