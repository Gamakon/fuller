# Evolving DAGs with GEP

> **Superseded (2026-10-09).** The current design is
> `docs/PLAN_population_dag.md` (§1, §2d): the population table holds two row
> kinds, ordinary chromosomes and homeotic chromosomes (population-wide shared
> definitions with a stable row id, a root type, memoised row values and a
> referrer count); an href names a homeotic row by its stable id, never a tail
> position; Fold is a variation operator that adds definitions, rewrites
> ordinary chromosomes to reference them and re-sorts the homeotic tail by
> use; execution follows a level schedule. FoldToDag did NOT replace the
> homeotic design: `docs/Geneframe_Homeotic_Genes_design.md` (revision 4, §8)
> is current, and §6 below is kept as history. This document's text is
> unchanged below.

Status: design discussion, no code written. Captures a conversation; supersedes
the framing in `Geneframe_Homeotic_Genes_design.md` where the two disagree (see
§6).

## 1. The motivating distinction (Neumann & Moerkotte)

`docs/PAPER_neumann_moerkotte_dag_query_plans.{pdf,txt}` (CSRD 2009) separates
two questions that are easy to conflate:

- **Share equivalence (≡S), §4.2**: two subexpressions compute *the same value*
  — same operator, same literals, same bindings, up to a bijective rename of
  inputs. This is checkable from the expressions alone, before any cost model
  runs. It says a fold is *legal*.
- **Cost-based dominance, §4.4–4.5**: whether folding on a given ≡S opportunity
  is actually *cheaper* once sharing's effect on the whole plan's cost is
  accounted for. Neumann's sharpest result (§4.4, Fig. 5): DAGs have **no
  optimal substructure** — a locally cheaper non-shared plan can lose to a
  locally more expensive one that enables sharing and wins overall. ≡S alone
  never decides this; only comparing full candidate costs does.
- **Magic sets** (his evaluation section, Mumick/Finkelstein/Pirahesh/
  Ramakrishnan 1990): a transformation that synthesizes a new, derived,
  scoped relation *purely so that sharing which was invisible to the optimizer
  becomes checkable*. The payoff in his Query 2 (Fig. 9, 70% runtime cut) comes
  from deliberately choosing a plan shape that makes sharing visible — not
  from sharing being found for free.

This document works out what each of these three ideas becomes when the
"plan" is a GEP chromosome instead of a query plan, and settles on a different,
more general design than an earlier draft (`Geneframe_Homeotic_Genes_design.md`,
§6).

## 2. Two searches, not one, and how they're related

An earlier framing of this problem ran two independent searches over a
chromosome's genes and almost treated them as alternatives to choose between.
They are not alternatives — they operate at different levels of generality,
on different schedules, and the first **feeds** the second.

### 2.1 Field-exact share equivalence (the supercompiler pass)

Match subtrees **exactly**: same op, same literal, same variable. No
generalization. This is Neumann's ≡S directly — `Sub(ProtectedSqrt(-87), x_0)`
occurring three times in one individual's eight genes is the same computation
three times over, not three similar-looking computations.

Measured (one-off study, `examples/gene_share_study.rs`, real 60s SR fits,
`n_genes=8 head=64 seed=7013`, feynman_I_6_2a and feynman_I_43_43): 108/800 and
147/800 individuals respectively have at least one such exact repeat within
their own genes; the top repeat in each case recurs up to 3× in a single
individual, and is itself seen across 79 and 124 individuals population-wide.
Total redundant op-evaluations (the `(count-1) × op_count` saved by computing
each repeat once): 583 and 442 across the two populations.

This pass:

- runs **per chromosome**, at expression/decode time, before evaluation;
- costs nothing to estimate — the savings number is pure AST analysis, no
  evaluation needed, which is exactly the "estimate the value of the
  optimisation before running" property that makes it worth doing early and
  often;
- is a correctness-preserving rewrite: folding `A ≡S A` into one shared
  definition changes nothing about what the chromosome computes, only how
  many times it's computed.

### 2.2 Function-type generalization (the kingdom-evolution pass)

Generalize every terminal (a `Var N`, a `Num` literal) to its `Ty`, keep every
internal op name concrete, and look for recurring **shapes** — `Sub(ProtectedSqrt(-87), x_0)`
and `Sub(ProtectedSqrt(69), x_3)` both generalize to `Sub(ProtectedSqrt(F), F)`.
This is weaker than ≡S on any single pair (it's not claiming the same value is
computed, only the same combination of operators), but it answers a different
question: does this *pattern* of combined symbols recur often enough, across
independently-evolved individuals, to be worth a name of its own in the
kingdom's vocabulary?

Measured on the same two populations, filtered to combinations of ≥2 existing
symbols (a single op over a terminal, e.g. `Neg(F)`, is normal use of an
existing symbol, not a discovery): 170/800 and 167/800 individuals contain a
≥2-op combination that recurs elsewhere in the population; top hits
`Sub(ProtectedSqrt(F), F)` (109×) and `ProtectedInv(Neg(F))` (134×).

This pass:

- runs **across the population**, periodically — it is a learning signal, not
  a per-evaluation cost optimization, so it doesn't need to run every
  generation (closer to the replacement dictionary's nursery-cohort cadence,
  `replacement-dictionary-learning-across-runs` in project memory);
- cannot be validated by AST analysis alone — "is this combination worth a
  named symbol" is ultimately an HFF-tournament question (do chromosomes that
  use it win?), not a static one;
- is where the two passes connect: §2.2 doesn't start a second independent
  search, it is the aggregation/promotion step **over** what §2.1 stages (see
  §4).

## 3. `FoldToDag`: the general operator

Both measured passes above were read-only studies. The operator that acts on
their findings:

```
FoldToDag(KarvaChromosome, SymbolTable, MagicSymbolTable)
    -> edits to (KarvaChromosome, MagicSymbolTable)
```

Note the signature: the output is **edits to things that already exist and
already evolve**, not a new DAG data structure. A chromosome stays a sequence
of karva-encoded genes; `MagicSymbolTable` stays a table in the same shape as
`SymbolTable` (`src/geneframe.rs`). `FoldToDag` never introduces a third
representation — the "DAG" is a semantic fact about the chromosome (some of
its subtrees are shared), not a different object the chromosome gets
converted into.

### 3.1 What `MagicSymbolTable` is

Named after Neumann's magic sets for the same reason his transformation
exists: it is a staging object that makes sharing checkable, scoped exactly
as narrow as what was actually found. A `MagicSymbolTable` entry is seeded
**field-exact** (§2.1) — it carries the literal `-87`, the variable `x_0`,
whatever the discovered computation actually was. This makes it, on its own,
"a less good general function" (Andrew's phrasing): useful only to the
individual(s) whose chromosome contains exactly that computation, not a
general-purpose primitive.

Generalization by type (§2.2) is the **promotion test**, not a separate
discovery mechanism: take a magic entry, replace its literals/variables with
typed slots, and check whether other independently-discovered magic entries
collapse to the same generalized shape. Enough independent support for one
shape is evidence the pattern is doing real, repeated work across distinct
individuals — not just saving one chromosome's eval cost once — and *that* is
what earns a row in the real kingdom `SymbolTable`, with its own typed arity,
available to every future individual.

So the full pipeline, end to end:

1. field-exact discovery, per chromosome, cheap, cost-justified immediately
   (§2.1) →
2. stage as `MagicSymbolTable` entries, literals and variables intact →
3. generalize staged entries by type, look for independently-recurring shapes
   (§2.2) →
4. a shape with enough independent magic-entry support is a kingdom-function
   candidate →
5. (open question, §5) HFF-tournament evidence before the candidate actually
   graduates into `SymbolTable` proper.

### 3.2 Finding candidates: generate everything, rank, don't choose

A chromosome's genes, asserted as roots into one `EGraph` built from the
kingdom's datatype, give exact subtree identity for free via egglog's
hash-consing at assertion time — no saturation needed for the field-exact
pass; both measured runs above found **zero** additional matches from a
bounded algebra+powers saturation beyond plain string identity, so on this
evidence the cheap version already captures everything real (this should be
re-checked on a kingdom whose saturation can introduce real identities the
raw AST can't show, e.g. `Add(x,x)` vs `Mul(2,x)` — neither test case
happened to exercise that).

There will usually be more than one maximal shared subtree in a chromosome —
fuller should not pick one fold to try. Per the project's "generate
everything, rank, never choose" discipline, every candidate fold is an
additional candidate individual (as `extract.rs`'s `denoise_variants`/
`extract_variants` already do for simplification), scored and let compete,
not selected by a heuristic in the fold operator itself.

## 4. Scoring folds: the new HFF dimension (future work)

Today's HFF scores a chromosome's *linked output* — accuracy, generalization,
etc. It has no way to see "how much does this fold save," because that is a
structural property of the whole chromosome (how many genes reference the
shared subtree, how many ops each reference saves), not a property of one
expressed tree's predictions.

This needs a **chromosome-scoped cost-estimating HFF dimension**, new work,
confirmed as such and deliberately deferred past this document: an objective
that can see across all of a chromosome's genes at once (not just the one
tree HFF currently scores), fed from the fold candidate's
`(count-1) × op_count` savings estimate — computable before any evaluation
runs, which is the "estimate the value of the optimisation even before
running" property §2.1 already has. Candidate folds would be sorted and
selected by this dimension exactly as every other HFF objective is: the
scoring chooses, the fold operator doesn't.

## 5. Re-encoding: the edited chromosome must stay evolvable

A fold shortens whichever gene(s) contained the now-shared subtree inline
(their expressed tree lost nodes, replaced by one reference). GEP's structural
invariant (`karva.rs`) is that every gene in a chromosome keeps a fixed
`head_length`, and `tail_len = head_len * (max_arity - 1) + 1` derived from
it — every gene must stay that length for crossover/mutation operators that
assume uniform gene width to keep working.

Checked directly against phylu's source (not guessed): phylu already has a
**fold** mechanism, but it turns out to answer a different question than this
document's `FoldToDag` — worth stating precisely, since it is tempting to
assume more precedent exists than actually does.

**What phylu's `fold` actually is** (`Engine::fold_winners`,
`src/evolve/engine.rs:5518`, run on the pump's fold beat,
`generation % fold_every == 0`, `engine.rs:7679`): it decodes one gene, finds
the biggest subtree whose device-measured output is constant across the
training rows (testing the *child's* variation, not the subtree's own
magnitude — `engine.rs:5640-5659`), and **constant-folds** it: collapses that
subtree to a single literal, the mean of its value over the rows
(`engine.rs:5695`). This is ordinary constant folding, scoped to one gene, no
cross-gene bookkeeping, no shared reference, no registry of multiple sites
pointing at one definition. It is not a DAG operator and is not precedent for
the sharing/refcount layer `FoldToDag` needs — that layer does not exist in
phylu today and would be new work.

**What phylu's fold *does* reuse, and `FoldToDag` can reuse too — the
re-encode mechanism.** This part is exactly the reversible machinery this
document assumed: `fold_winners` builds a single `Graft { token: rnc_id,
kids:[0,0], dc: slot }` — a `?` constant leaf (`engine.rs:5730`) — and calls
`vary::relevel(tokens, layout, vhead, codes, &tree, &[(pos, GRAFT)], &leaf)`
(`engine.rs:5733`, implementation `src/evolve/vary.rs:286`), which rebuilds
the gene's head+tail by a level-order BFS walk, substitutes the graft at
`pos`, and drops the collapsed subtree. `relevel` enforces the karva
invariants itself and returns `Unfit` (`HeadOversize`/`DcOversize`/
`NotClosed`, `vary.rs:329-334`) if the edit won't close — a refused relevel
leaves the gene byte-identical (`engine.rs:5745-5750`), so a caller never has
to separately validate the result.

**The padding question, answered**: `relevel` writes the new, shorter token
sequence into `tokens[..m]` and the new Dc into `tokens[ht..ht+k]`, and
*leaves `tokens[m..]` untouched* (`vary.rs:335-336`) — the freed positions
keep whatever bytes they had before. They go non-coding (the level-order
decode closes the tree before reaching them), but they are **not** permanent
dead weight: phylu's variation kernel mutates the *whole* fixed gene width
every generation regardless of what currently expresses
(`for i in 0..row_w` where `row_w = n_genes*(head+2*tail)`, `vary.wgsl:539`),
and crossover/transposition act over the full width too. So a buried,
currently-non-coding position can be mutated and later pulled back into the
expressed region by a subsequent edit — reclaimable, not inert. This is
exactly the property this document's §5 was designing for, confirmed as
already true of phylu's gene representation generically, independent of fold.

**One more mechanic worth carrying over**: `fold_winners` does not edit a
chromosome in place. It writes the cleaned gene into the island's *worst* row
as a new, unevaluated candidate sitting beside the untouched original
(`engine.rs:5753-5776`), and lets the HFF tournament judge it — never
asserting the edit is good, only that it is a candidate. `FoldToDag` should
do the same: a fold's winning candidate enters the population as one more
individual to be scored, exactly as `extract.rs`'s `denoise_variants` already
does for simplification, not as an in-place rewrite the operator decides is
correct.

**What is genuinely new, not reused**: the cross-gene (or cross-chromosome)
sharing layer itself — detecting that the *same* subtree occurs at more than
one site, representing a definition with more than one reference, and
re-encoding every referencing site (not just one collapsed site) consistently.
`relevel`'s single-graft-site API handles one `(pos, GRAFT)` substitution at a
time; `FoldToDag` needs to either call it once per referencing site (simplest,
assuming each site's substitution is independent) or extend it to accept
multiple simultaneous substitutions sharing one underlying definition — an
open implementation question, not a precedent phylu already answers.

## 6. Where this supersedes the homeotic-gene design

*Note (2026-10-09):* this section's claim is reversed. The homeotic design,
made population-wide in `Geneframe_Homeotic_Genes_design.md` §8 and
`PLAN_population_dag.md` §1, is current; FoldToDag did not replace it.

`Geneframe_Homeotic_Genes_design.md` proposed `REF_0..REF_{G-1}` — a fixed
terminal family where gene *i* may only reference gene *j < i*. That design is
too narrow for what this document describes:

- it is wired to one fixed chromosome shape (gene-index order as the only
  topological order) rather than any discovered sharing, anywhere, within or
  across genes;
- its "no new symbol table" simplification was a virtue when the only goal was
  cross-gene linking, but it is the wrong simplification once the real goal is
  **evolving the kingdom's vocabulary itself** — a `MagicSymbolTable` staging
  new, typed, potentially-promotable symbols is a different, larger idea than
  a fixed per-chromosome reference family;
- `REF_j`'s acyclicity-by-construction (just gene index order) was a
  convenience, not a requirement — `FoldToDag`'s sharing can be found in any
  DAG-consistent order via the e-graph's own structure, without needing a
  fixed reference-direction rule baked into the symbol table.

The homeotic design's useful, still-valid observations (karva round-trip is
bidirectional and untouched by adding a new terminal kind; `GeneLinker`'s
subset-exclusion already handles leaving a hoisted gene out of the top-level
link) carry over. Its core mechanism (§3.1–3.3 of that document) does not —
`FoldToDag` + `MagicSymbolTable` is the design going forward.

## 7. Summary of the pipeline

1. phylu evolves a chromosome: a sequence of karva genes, each a multi-typed
   AST once decoded.
2. At expression time, fuller runs field-exact share-equivalence detection
   (§2.1, §3.2) across the *whole* chromosome's trees, not gene-by-gene in
   isolation — finds every exact repeated subtree, generates every candidate
   fold, picks none of them itself.
3. A chromosome-scoped cost-estimating HFF dimension (§4, future work) scores
   each candidate fold's savings; candidates are sorted and selected the way
   every other HFF dimension already is.
4. The winning fold is staged as a `MagicSymbolTable` entry — field-exact,
   with its typed signature recorded alongside (§3.1).
5. The chromosome is rewritten: occurrences of the shared subtree become
   references to the new magic symbol, with the real literals/variables now
   carried as that symbol's arguments.
6. The edited gene(s) are re-encoded to fixed-length karva, padded (§5,
   mechanics pending from phylu) so they stay evolvable, not inert.
7. Separately, periodically, across the population: staged `MagicSymbolTable`
   entries are generalized by type (§2.2) and checked for independent,
   recurring support; a shape with enough support is a kingdom-function
   candidate for eventual promotion into `SymbolTable` proper (§3.1, pending
   §5's open question on what gate — shape-count alone, or also HFF-tournament
   evidence — decides a graduation).
