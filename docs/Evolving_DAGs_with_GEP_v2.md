# Evolving DAGs with GEP (v2)

## Context

**Objective.** Let a GEP chromosome reuse a computation it repeats, instead
of recomputing it at every site, and let computations that keep recurring
across many individuals become new named functions in the kingdom's own
vocabulary.

**Approach.** Two searches over a chromosome's genes, run at different times
for different reasons, feeding one pipeline: find exact repeated
subexpressions cheaply, stage them, and let the ones that generalize across
many chromosomes earn a permanent symbol.

**Unique feature.** The kingdom's symbol table is normally fixed at compile
time. This lets it grow from what evolution actually discovers, without ever
treating a discovery as true before the HFF tournament says so.

**Status.** Design discussion, no code written. v1 of this document
(`Evolving_DAGs_with_GEP.md`) covers the same ground; this version replaces it
as the one to build from. The two disagree on the homeotic-gene design
(`Geneframe_Homeotic_Genes_design.md`) — v2's view, like v1's, is that
`FoldToDag` replaces it; see Open questions.

**Correction (2026-10-09).** FoldToDag did NOT replace the homeotic design.
The current design is `docs/PLAN_population_dag.md` (§1, §2d), stated in
`Geneframe_Homeotic_Genes_design.md` revision 4, §8, which is current over
this document where they differ. The genotype gains a second row kind: the
population table holds ordinary chromosomes and homeotic chromosomes
(population-wide shared definitions with a stable row id, a root type,
memoised row values and a referrer count). An href names a homeotic row by
its stable id, never a tail position. Fold is a variation operator that adds
homeotic rows, rewrites ordinary chromosomes to reference them (value-
preserving, so the rewrite is in place, not a second candidate) and re-sorts
the homeotic tail by use. Execution follows a level schedule. The two
equivalences of §1, the evidence of §5 and the `relevel` mechanics of §4
remain true.

**Why.** DAG-shaped reuse of subcomputations is Thomas Neumann's lifelong
subject in query optimization (Neumann & Moerkotte, CSRD 2009,
`docs/PAPER_neumann_moerkotte_dag_query_plans.pdf`), and the same two ideas he
separates there — is a fold legal, and is it worth it — turn out to map
cleanly onto GEP chromosomes once "query plan" is read as "chromosome."

## 1. Two kinds of equivalence

Neumann's paper keeps two questions apart that are easy to blur:

- **Is a fold legal?** Two subexpressions are *share equivalent* if they
  compute the exact same value: same operator, same literals, same
  variables. This can be checked from the expressions alone — no execution
  needed.
- **Is a fold worth it?** Share equivalence only says folding is allowed. His
  sharpest finding is that DAGs have no optimal substructure: a plan that
  looks locally worse can still win once its sharing is accounted for, so
  cost has to be compared on whole candidates, never decided from legality
  alone.

GEP chromosomes need both tests too, but at two different scopes:

**Function-field equivalence.** The user's name for an exact match: same op,
same literal, same variable, found inside one chromosome's own genes. If one
individual computes `Sub(ProtectedSqrt(-87), x_0)` three times across its
eight genes, that is the same value computed three times, and sharing it is
a pure execution-cost win: compute it once, reuse the result everywhere it
was needed. This is a supercompiler optimization — legal by construction,
and its savings can be estimated before the chromosome is ever evaluated,
because it's a property of the AST, not of running it.

**Function-type equivalence.** The user's name for a generalized match: every
variable and literal replaced by its type, every operator kept as-is. Under
this rule, `Sub(ProtectedSqrt(-87), x_0)` and `Sub(ProtectedSqrt(69), x_3)`
are the same shape, `Sub(ProtectedSqrt(F), F)`, even though neither computes
the other's value. This cannot say a fold is free the way function-field
equivalence can; it says a *pattern* of operators keeps recurring across
independently-evolved individuals, which is evidence worth having a name for
it, not evidence of a safe execution-cost win.

## 2. `FoldToDag`

```
FoldToDag(chromosome, symbol_table, magic_table) -> (chromosome', magic_table')
```

`chromosome` is the sequence of karva-encoded genes being edited.
`symbol_table` is the kingdom's existing, fixed vocabulary, read only.
`magic_table` is the staging table described in §3. The result is an edited
chromosome and an updated staging table — never a new data structure to
represent the DAG. The DAG is a fact about the chromosome (some of its
subtrees now share one definition), not an object it gets converted into.

### 2.1 Finding candidates

Assert every gene's decoded tree as a separate root into one e-graph built
from the kingdom's datatype. Egglog hash-conses identical subtrees at
assertion time, so exact (function-field) matches are free: no rewrite rules
need to run. A bounded algebra-and-powers saturation pass was tried on top of
this and found zero additional matches on the two laws measured below, so the
cheap, rule-free version is the default; a kingdom whose algebraic identities
can equate two syntactically different subtrees (`Add(x,x)` and `Mul(2,x)`,
for instance) should be re-checked before assuming that holds generally.

A chromosome will usually contain more than one shared subtree. `FoldToDag`
does not choose between them. Every candidate fold becomes one additional
individual in the population, the same way `extract.rs`'s
`denoise_variants`/`extract_variants` already turn every simplification
candidate into something the tournament scores, rather than something a
heuristic picks.

### 2.2 Scoring candidates

Today's HFF scores one chromosome's output: accuracy, generalization, and so
on. It has no dimension for "how much does folding this subtree save,"
because that is a property of the whole chromosome — how many genes
reference the shared subtree, how many operators each reference removes —
not a property of one expressed tree's predictions.

A new, chromosome-scoped cost dimension is needed to score fold candidates:
one that reads across every gene in the chromosome at once, not just the
gene being folded, fed from `(occurrences − 1) × ops_per_occurrence` as a
pre-computed estimate. This is new work, not yet built, and is out of scope
for this document beyond naming it as the next piece.

### 2.3 Staging and rewriting

The winning candidate is staged as a new entry in `magic_table` (§3), and the
chromosome is rewritten: every site where the shared subtree occurred is
replaced by a reference to that entry, carrying the occurrence's own
literals and variables as arguments. This is the one structural requirement
`relevel`'s existing API doesn't yet cover on its own — see §4.

## 3. The magic symbol table

Named for Neumann's magic sets, which solve a related problem: synthesizing a
new, scoped object specifically so that sharing which was otherwise invisible
becomes checkable. `magic_table` plays that role for a kingdom's vocabulary:
a place where a real, found computation can be named before anyone decides
whether it deserves to be a permanent symbol.

Each entry holds:

| field | what it is |
|---|---|
| `symbol` | an ordinary `Symbol` row (`src/geneframe.rs:140`): kingdom, id, name, alias, semantic id, arity — exactly the shape a `SymbolTable` row already has |
| `definition` | the concrete subtree this entry was found as, as a `Math` s-expression, literals and variables intact |
| `source` | which chromosome and which gene positions it was found at |
| `occurrences` | how many times it recurred, at staging time |

An entry is seeded function-field: its `definition` is one specific found
computation, not a general-purpose primitive. That is deliberate — it only
has to be real, not good on its own. Whether it generalizes is decided
separately, in §3.1.

### 3.1 Promotion

Take every staged entry's `definition`, replace its literals and variables
with their types (function-type equivalence, §1), and group entries by the
resulting shape. A shape with enough independently-discovered entries behind
it, found in separate chromosomes rather than one chromosome's repeats, is a
candidate for promotion: a new row in the kingdom's real `symbol_table`, with
its own typed arity, available to every future individual. What counts as
"enough" — a count threshold alone, or also requiring that chromosomes using
the candidate win in the HFF tournament before it graduates — is an open
question (§6).

## 4. Re-encoding the chromosome

Every gene in a chromosome keeps a fixed head length, with the tail length
derived from it (`karva.rs`), so that crossover and mutation can assume
uniform gene width. Folding a subtree out of a gene shortens what that gene
expresses, and the edit has to land back in a fixed-length token sequence.

phylu already has a mechanism for exactly this kind of edit, used by its
existing `fold` operator (`Engine::fold_winners`, `engine.rs:5518`) for a
different purpose: collapsing a subtree whose value is constant across the
training data down to a single literal. That purpose is constant folding,
not sharing — phylu's fold has no notion of one definition referenced from
more than one site, and nothing resembling `magic_table` — but the mechanism
it uses to perform the edit is exactly what `FoldToDag` needs:

- `fold_winners` builds one `Graft { token, kids, dc }` — a constant leaf —
  and calls `vary::relevel(tokens, layout, vhead, codes, &tree, &[(pos,
  GRAFT)], &leaf)` (`engine.rs:5733`, `vary.rs:286`).
- `relevel` rebuilds the gene's head and tail with a level-order walk,
  substituting the graft at `pos` and dropping the subtree it replaced. It
  checks the karva invariants itself and returns `Unfit` if the result
  wouldn't close (`vary.rs:329`), leaving the gene untouched on failure
  (`engine.rs:5745`).
- The freed positions after a successful relevel are left as whatever bytes
  were already there (`vary.rs:335`). They stop being expressed, but they are
  not dead: phylu's mutation and crossover operators act over every gene's
  full fixed width regardless of what currently decodes (`vary.wgsl:539`), so
  a buried position can be mutated and pulled back into the expressed region
  later. No special filler is needed; leaving the old bytes in place and
  letting the existing operators keep touching the whole width is sufficient.
- `fold_winners` never edits in place. The folded gene is written into the
  population as a new, unevaluated candidate next to the untouched original,
  and the tournament decides which survives (`engine.rs:5753`). `FoldToDag`
  should do the same.
  *Correction (2026-10-09):* true of the constant fold, not of the share
  Fold. A share fold is value-preserving, so Fold rewrites ordinary
  chromosomes in place to reference homeotic rows (`PLAN_population_dag.md`
  §1).

What `relevel` does not yet cover: a magic-symbol reference needs the shared
definition's free leaves carried along as live argument slots at the
reference site, and `relevel`'s current API substitutes one graft at one
position at a time. Whether `FoldToDag` calls it once per referencing site
or needs an extended version that substitutes several sites against one
shared definition in a single pass is open (§6).

## 5. Evidence

A one-off study (`examples/gene_share_study.rs`) ran both equivalence checks
on two real 60-second SR fits (`n_genes=8`, `head=64`, seed 7013): Feynman
I.6.2a (one input variable) and Feynman I.43.43 (four input variables), 800
individuals each.

| | I.6.2a | I.43.43 |
|---|---|---|
| individuals with ≥1 function-field repeat within their own genes | 108/800 | 147/800 |
| highest repeat count seen in a single individual | 3 | 3 |
| total redundant operator-evaluations saved across the population | 583 | 442 |
| individuals containing a ≥2-operator combination that recurs elsewhere in the population (function-type match) | 170/800 | 167/800 |
| top recurring combination | `Sub(ProtectedSqrt(F), F)`, 109 individuals | `ProtectedInv(Neg(F))`, 134 individuals |

Reading the first row: on I.6.2a, 108 of 800 evolved chromosomes compute at
least one subexpression more than once within their own eight genes — real,
immediately actionable sharing, not a hypothetical case. The bounded
algebra-and-powers saturation pass found no matches beyond plain identity on
either law (§2.1).

## 6. Open questions

1. Does a fold candidate's score come from the count-threshold alone (§3.1),
   or does promotion into `symbol_table` also require chromosomes using the
   candidate to win HFF tournaments, as separate evidence from the staging
   count?
2. Does `relevel` need an extended, multi-site form to rewrite several
   reference sites against one shared `magic_table` definition in a single
   pass, or is calling it once per site sufficient (§4)?
3. Does the zero-additional-matches result from saturation (§2.1) hold on a
   kingdom whose algebraic identities can equate syntactically different
   subtrees, or was that just true of the two laws tested?
4. `Geneframe_Homeotic_Genes_design.md`'s `REF_j` mechanism assumed a fixed
   per-chromosome reference direction (gene *i* may reference only gene
   *j < i*) and no new symbol table. *Answered 2026-10-09: no, it does not
   replace it; the homeotic design, made population-wide, is current
   (`PLAN_population_dag.md` §1).* `FoldToDag` + `magic_table` replaces
   that design rather than extending it — confirm this is the intended
   direction before any of its still-useful observations (karva round-trip
   stays bidirectional; `GeneLinker`'s subset exclusion already handles
   leaving a folded gene out of the top-level link) get carried forward into
   new code.
