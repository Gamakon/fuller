# Saturated share-equivalence: finding `exp(x+y)` ≡ `exp(x)*exp(y)`

## Context

**Objective.** `FoldToDag` (`Evolving_DAGs_with_GEP_v3.md`) finds and folds
*function-field* equivalence: exact structural repeats, found by egglog's
hash-consing with no rewrite rule firing. It is built, tested, and measured
(`docs/PLAN_fold_to_dag_generef.md`; `phylu/examples/fold_to_dag_measure.rs`).
This document is about the next tier up: two subexpressions that compute the
same value but are *not* written identically — `exp(x+y)` versus
`exp(x)*exp(y)` — found only by saturating with rewrite rules first. This is
detection design, not yet code.

**Approach.** Assert every gene of a chromosome into one shared e-graph,
saturate it with one of fuller's existing bounded rule families, then run a
DAG-cost extractor over all gene roots jointly to pick one concrete form per
e-class — the extractor is what *decides* that `exp(x)*exp(y)` is the shared
form, not `exp(x+y)`. The existing `maximal_shared` site/containment code
then runs unchanged on the extracted result. Folding is gated by measured
cost, not op-count, with a tree-extraction fallback when the economics don't
work out.

**Unique feature.** The sequence below reuses every piece already built —
the same egraph assert, the same bounded rulesets, the same site-based
maximal-match code `FoldToDag` already has tested — and adds exactly one new
component (a DAG-cost extractor) plus one new decision layer (fold only if
the real, measured per-read and dispatch costs are covered). Nothing here
requires a new representation or a new rule family.

**Status.** Design only, arrived at through several rounds of review that
each found a real flaw in the previous round (recorded in §4). Not yet
implemented, not yet estimated against real data.

**Why this document exists.** The question "can fuller find equivalence
across *rewritten* forms, not just exact repeats" came up directly after
`FoldToDag`'s real measurement landed (§5 of this doc). The first four
proposed sequences for answering it were each wrong in a specific, named way
— this document exists so the final sequence, and the reasoning that
eliminated the wrong ones, survives past this conversation.

## 1. The question

`FoldToDag` only finds matches that are already written the same way.
Today's detection (`fuller::extract::maximal_shared`) asserts each gene as a
named root into an egraph and relies on hash-consing at assertion time: two
nodes merge only if they are already structurally identical (same op, same
literal, same variable). No rewrite rule runs. This is why it is called
*function-field* equivalence, and why it is cheap and exact.

It is also why `exp(x+y)` in one gene and `exp(x)*exp(y)` in another are
invisible to it — they are never asserted as the same node, so hash-consing
never merges them. Finding that equivalence requires actually proving
`exp(a+b) = exp(a)*exp(b)`, which is a rewrite rule, not an assertion-time
coincidence. The question this document answers: can fuller's existing
egglog substrate do that, across a whole chromosome, and still produce
something `FoldToDag`'s fold step can act on?

## 2. What survives unchanged from `FoldToDag`

- The multi-root assert: one `let` per gene into an `EGraph`.
- Bounded, kill-guarded saturation using one of the existing rule families
  (algebra+powers, the same subset `denoise` already trusts — see
  `CLAUDE.md`'s non-confluence warning: rule families are NOT saturated
  together).
- `maximal_shared`'s site-based detection: DFS from each gene's root,
  record `(gene_index, path)`, group by shared node, sort largest-first,
  exclude a site that sits inside an already-accepted larger match. This
  logic is unmodified — it only needs a concrete, acyclic DAG to walk, and
  does not care whether a shared node arose from identical text or from
  rewriting.
- `fold_chromosome`'s GENEREF mechanism and the two-pass device chain
  (`eval_population_folded`), unmodified. Folding is still "replace every
  occurrence site with a reference to one extracted definition."

## 3. What is new: extraction between saturation and detection

### 3.1 Why `maximal_shared` cannot run directly on a saturated e-graph

A saturated e-graph is not a DAG you can walk. Each e-class is a *set* of
equivalent e-nodes (that is the entire point of saturation), and the
rewritten e-graph contains cycles that commutative/associative rules
introduce (e.g. `a*b` and `b*a` can each sit in e-classes that reference each
other). Running a DFS-from-root site collection over that structure has no
well-defined answer: at every e-class visited, which child set do you
descend into? Pick differently in two different genes and you can miss the
very match you were looking for.

### 3.2 The extractor's job

A DAG-cost extractor (e.g. extraction-gym's greedy-DAG, or ILP) picks
exactly ONE e-node per e-class, globally, across every gene's root jointly —
not per gene, not per e-class in isolation. "Jointly" is what makes it
possible for the extractor to choose `exp(x)*exp(y)` in BOTH genes
consistently, rather than an arbitrary (and possibly different) choice in
each. The output is a concrete, acyclic DAG: no e-class ambiguity, no
cycles. `maximal_shared`'s existing site-collection code can walk it exactly
as it walks today's un-saturated `serialize()` output.

### 3.3 Thomas Neumann's actual contribution here

Neumann & Moerkotte's separation of "is a fold legal" from "is a fold worth
it" ([`PAPER_neumann_moerkotte_dag_query_plans.pdf`]) is the structural
lesson this design borrows, not a specific algorithm to port. Legality is
what saturation plus hash-consing establishes — two subexpressions provably
compute the same value. Worth-it is a separate, cost-based decision layered
on top, made only after legality is known. The design history below (§4) is
mostly the story of that separation being violated and then restored.

## 4. Design history: four wrong sequences, in order, and why

Recorded because each wrong version is a specific, instructive mistake, not
a strawman.

**Attempt 1 (mine): run `maximal_shared` directly on the saturated
e-graph's `serialize()` output.** Wrong per §3.1 — a saturated e-graph is
not a walkable DAG. *Fix: insert a DAG-cost extractor between saturation and
detection (first advisor correction).*

**Attempt 2 (mine): give the extractor a cost function that adds a GENEREF
cost to any e-node used by ≥2 sites.** Wrong because this is circular:
whether a node ends up used by ≥2 sites is the *output* of extraction (and
of the later maximal-match step), not something known beforehand. Standard
DAG extractors (greedy-DAG) also structurally assume per-e-node cost is
fixed and reuse is free — they cannot take a cost that depends on their own
outcome; that requires ILP with an explicit sharing variable per e-class, a
materially bigger piece of machinery than this problem calls for. *Fix: let
the extractor assume reuse is free (its native assumption, satisfied by
plain greedy-DAG), and apply the real per-read/dispatch costs as a separate
decision AFTER extraction, never inside it (second advisor correction).*

**Attempt 3 (mine): decide-per-node-then-decide-per-chromosome, in that
order, running maximal-match last.** Wrong in three ways, each fixed by the
same correction round:

- Pricing shared nodes before running the maximal-match/containment step
  prices nested shares that containment would have removed anyway — wasted
  work on numbers that get thrown away.
- "3.6% slower" (`FoldToDag`'s real measured population-wide result) is a
  *net* figure — savings minus every overhead combined — not the fixed
  per-dispatch cost in isolation. Using it directly as a gate threshold
  conflates two different numbers.
- A shared node that CONTAINS another already-folded node would have its
  subtree cost counted including the folded child's cost, double-counting
  that child's saving.

*Fix, all three at once (third advisor correction): run maximal-match
immediately after extraction, before any cost decision, so pricing only
ever touches the final, containment-resolved match list; calibrate the
fixed second-pass dispatch cost once (time pass 2 with an empty definitions
table), not per chromosome; when computing a shared node's subtree cost,
exclude the cost of any child that is itself already a folded reference —
which requires deciding nodes bottom-up, innermost first, so a child's own
decision exists before its parent's cost is computed (fourth advisor
correction, below).*

**Attempt 4 (mine): a sequence whose step 7 described a convergence loop
that no numbered step actually performed, with no ordering constraint on
step 5's bottom-up requirement, and a calibration step that read as
per-chromosome.** Three omissions, not a wrong idea: the block-and-re-extract
action needed to be an explicit step (not just referenced from a later
prose paragraph), step 5 needed its bottom-up order stated as a
requirement rather than implied, and step 4 needed to say plainly that it
runs once per device/run and is reused, not recomputed per chromosome.
*Fix (fourth advisor correction): see §5 below, which is the corrected
sequence.*

## 5. The sequence, as it now stands

1. **Assert all genes into one e-graph, saturate** with one bounded rule
   family — unchanged from `FoldToDag`.

2. **DAG-extract with free reuse** (plain greedy-DAG / extraction-gym, no
   custom cost function). This is the step that picks `exp(x)*exp(y)` over
   `exp(x+y)` when sharing makes the joint form cheaper. It assumes reuse is
   free, which is wrong in general but is exactly the assumption that keeps
   this step non-circular (§4, attempt 2) — the correction for that
   assumption happens later, in steps 5-6, not here.

3. **Run `maximal_shared`'s existing site/containment code on the extracted
   DAG**, immediately — before any cost decision. This produces the final
   list of shared nodes and their real, post-containment use counts. Nested
   shares that containment would reject are already gone by the time any
   pricing happens.

4. **Calibrate the fixed second-pass dispatch cost once** — e.g. time
   `eval_pass_with_shared` with an empty definitions table, once per device
   or per run, and reuse that number everywhere below. This is NOT a
   per-chromosome step: it never derives from a net before/after comparison
   on any one chromosome, and it is not re-measured per chromosome either.

5. **Per shared node, BOTTOM-UP** (innermost shares first, from step 3's
   list): fold only if `(uses - 1) × subtree_cost > uses × per_read_cost`,
   where `subtree_cost` EXCLUDES the cost of any child that is itself
   already a folded reference. The bottom-up order is load-bearing, not a
   detail: "exclude an already-folded child's cost" is only a well-defined
   subtraction if that child's own fold decision was already made —
   deciding outer nodes before inner ones would exclude a cost that hasn't
   been decided yet.

6. **Block-and-re-extract, inline with step 5**: if step 5 rejects folding
   a node, AND that node's concrete form was chosen by the step 2 extractor
   specifically because it assumed sharing was free (i.e. a different,
   non-shared form would have been chosen under honest per-use cost), block
   that e-node in the extractor and re-run step 2. Re-run steps 3 and 5 on
   the new extraction. At most two rounds; this is the mechanism that
   discards a rewritten form (like `exp(x)*exp(y)`) that was only
   attractive under the false free-reuse assumption, rather than leaving it
   half-adopted in the final program.

7. **Chromosome-level gate**: once step 5/6 has settled (accepted, or two
   rounds exhausted), fold this chromosome only if the sum of accepted
   savings exceeds the calibrated dispatch cost (step 4).

8. **Fallback**: if a chromosome fails the step 7 gate, or step 6's
   block-and-re-extract loop has not settled after two rounds, use ordinary
   tree extraction for that chromosome instead — the mechanism never
   produces a worse-than-baseline result, it just declines to fold.

9. **Write GENEREFs back** into whichever chromosomes passed the gate, via
   `fold_chromosome`, unchanged.

## 6. What is not yet known

- No cost numbers exist yet for steps 4-7 on real saturated chromosomes —
  this document is the design, not a measurement. The pattern from
  `FoldToDag`'s own history (a plausible-sounding mechanism that turned out
  3.6% slower net, for reasons only visible once actually measured) applies
  here at least as strongly, since this adds an extraction and a
  convergence loop on top of the same two-pass dispatch cost.
- Which bounded rule family to saturate with is not decided per se — it
  inherits `denoise`'s existing algebra+powers subset as the default
  starting point, per `CLAUDE.md`'s non-confluence warning, but has not been
  checked against the specific rewrites needed for `exp(a+b) = exp(a)*exp(b)`
  — confirming that rule exists (or adding it) in the relevant ruleset file
  is unverified at the time of writing.
- Whether extraction-gym is vendored, available, or needs adding as a
  dependency has not been checked.
- Magic sets (Neumann & Moerkotte's technique for making partially-scoped
  sharing visible — e.g. the same subexpression computed over different
  row-selections) are noted in §3.3's source material as a further
  extension, not part of this design.
