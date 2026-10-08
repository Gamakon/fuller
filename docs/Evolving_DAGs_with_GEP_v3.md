# Evolving DAGs with GEP

## Abstract

Gene expression programming evolves trees, but the programs those trees
express are, semantically, directed acyclic graphs: whenever two
subexpressions compute the same value, the tree form pays to compute it
twice while a DAG would compute it once. We show that evolved multi-gene
chromosomes already contain such repeated subcomputations at a rate worth
acting on, that an e-graph recovers them from the expressed program without
any search, and that collapsing them does two things at once: it is a pure
execution-cost win for the individual chromosome, and it produces a stream
of candidate functions which, once enough of them agree on shape across
independently-evolved individuals, are evidence for growing the function set
itself. We propose an operator, `FoldToDag`, and a staging table for the
candidates it finds, and set out what of the existing machinery survives
unchanged and what is new.

## 1. The problem

Gene expression programming should evolve programs whose shared
subcomputations are computed once, not once per occurrence.

A multi-gene chromosome in Ferreira's formulation is a sequence of
independently evolved trees, combined only after each has been evaluated in
full, by a linker over their finished values. No gene can see inside another
gene's tree, and nothing in the representation lets two genes — or two
places within one gene's own tree — share a subcomputation. If an individual
has discovered, by whatever combination of crossover and mutation produced
it, that the same subexpression is useful in three places, it pays to
evaluate that subexpression three times. The forest of trees a chromosome
expresses is exactly that: a forest, never a graph, by construction.

This is an old problem under a new name. Query optimizers faced it when a
plan referenced the same relation or the same subplan more than once, and
the standard treatment, due to Neumann and Moerkotte, separates two
questions that are easy to run together. The first is whether a fold is
*legal*: do two subexpressions compute the same value, so that replacing
both with one shared computation changes nothing about what the program
does. The second is whether a fold is *worth it*: once a shared computation
has more than one consumer, the cost of the whole program depends on the
sharing, not just on the pieces in isolation, and their sharpest finding is
that this breaks the usual assumption of optimal substructure — a locally
cheaper unshared plan can lose to a locally more expensive one that enables
sharing and wins overall. Legality is a static fact about the expressions;
worth is a comparison of whole candidates. Neither question answers the
other.

A GEP chromosome has both versions of this problem, and at two different
scopes. Within one individual, a repeated subexpression is a pure execution
cost, checkable and foldable without touching the search at all. Across a
population, independently-evolved individuals occasionally discover the
same combination of operators more than once, in different forms, which is
weaker evidence — not that any one value is repeated, but that a pattern of
operators is pulling its weight often enough to be worth naming. We treat
these as the two forms Neumann's legality question takes in this setting,
and we call them, respectively, function-field equivalence and function-type
equivalence.

## 2. The hypothesis

We hold four claims open to being wrong, and the remainder of this document
reports what we found when we checked them.

First, that evolved GEP chromosomes, run under ordinary symbolic regression
selection pressure with no instruction to reuse anything, already contain
subexpressions that occur more than once with identical operators, literals
and variables — function-field equivalence — at a rate worth acting on, not
as a rare curiosity.

Second, that these can be found by exact structural matching alone, without
running any rewrite rule: an e-graph's hash-consing at assertion time is
sufficient, because the matches we care about here are identity, not
algebraic equivalence.

Third, that collapsing a function-field match into one shared definition is
unconditionally a cheaper way to compute the same chromosome — no
accuracy is at stake, only the number of operations performed.

Fourth, that the same discoveries, generalised by replacing every literal and
variable with its type, recur across independently-evolved individuals often
enough that the recurrence itself is evidence worth feeding back into the
function set, as opposed to evidence that only ever concerns one individual.

We checked the first two claims directly. Two short evolutionary runs on
real symbolic-regression problems — one with a single input variable, one
with four — each produced several hundred individuals, between an eighth and
a fifth of which contained at least one subexpression repeated, verbatim,
within their own genes; running a bounded algebraic rewrite pass on top of
plain structural matching found nothing that matching alone had missed. The
full figures are in the appendix. We take this as enough to proceed to the
proposal; we have not yet built the apparatus to test the third and fourth
claims against a running tournament, and we say so plainly where that
matters below.

## 3. The available tools, and where each one stops short

Four pieces of existing machinery bear on this problem, and each one gets us
part of the way.

The karva genotype gives us a representation in which every individual,
after crossover or mutation of any kind, decodes to a syntactically valid
program — there is no such thing as an invalid offspring, because the
decoder reads a fixed-length string of tokens top-down and simply stops
early if a subtree closes before the string runs out. This guarantee is the
reason gene expression programming is usable at all under heavy structural
variation. But the guarantee is purchased at the level of one gene's own
tree: nothing about the decoding process, or the fixed-length invariant that
makes it work, has anything to say about two trees, or two places in one
tree, computing the same value.

A typed symbol table gives the search its vocabulary: a function's name,
its real-domain semantics, and a signature recording how many inputs of each
type it consumes and produces. A kingdom — symbolic regression, or a regular
expression dialect, or any other typed sublanguage the search is asked to
evolve — is simply a query selecting the rows that belong to it. This gets
us a vocabulary that can, in principle, grow without disturbing anything
already built on top of it, because adding a row is additive. What it does
not get us is any mechanism for a new row to arrive from evidence the search
itself produced; today every row is authored by hand, once, before the
search begins.

An e-graph gives us congruence closure for free: assert several expressions
into one graph, and any two subterms that are syntactically identical, or
that a rewrite rule proves equal, land in the same equivalence class
automatically, with no extra bookkeeping on our part. This is precisely the
mechanism that recovers sharing from an expressed program after the fact,
without any purpose-built search for it. What stops us here is a matter of
use, not of mechanism: every extraction path that turns an e-graph term back
into a program flattens it to a single tree and throws the sharing away,
because flattening is what every consumer downstream has so far needed.

A reversible gene-editing mechanism already exists, built for a different
purpose: collapsing a subtree whose measured value turns out to be constant
across the training data into a single literal. The edit it performs —
substitute a smaller subtree at some position, rebuild the gene's head and
tail by a level-order walk, and refuse the whole edit if the result cannot
close within the fixed length — is exactly the reversible, length-preserving
rewrite any collapsing operator over a karva gene would need. What it does
not do is know about more than one occurrence: it edits one position in one
gene, with no notion of a definition referenced from several sites at once.

A tournament, scoring each candidate on whatever dimensions a run declares
and comparing them on a common footing regardless of how many dimensions or
what their units are, gives us the only mechanism in this system that is
allowed to decide whether a candidate is good. What it does not yet have is
any dimension that can see a chromosome's structure as a whole: today it
scores a chromosome's expressed output, which has nothing to say about how
many times a subexpression it is built from gets evaluated.

Put together, the four tools already give us validity, vocabulary, sharing
recovery, and judgment. None of them, as built, connects sharing recovery to
the vocabulary or to judgment. That connection is the proposal.

## 4. The proposal

We propose an operator

```
FoldToDag(chromosome, symbol_table, magic_table) -> (chromosome', magic_table')
```

acting on a chromosome's karva-encoded genes, the kingdom's existing,
unmodified symbol table, and a new staging table described below. Its
result is an edited chromosome and an updated staging table, never a third
representation: the DAG this document is named for is a fact about which
subtrees a chromosome now shares, not an object the chromosome is converted
into.

### Finding and scoring candidates

Every gene in a chromosome is decoded and asserted as a separate root into
one e-graph built from the kingdom's datatype. Because an e-graph
hash-conses identical subterms at the point they are asserted, every
function-field match across the whole chromosome's genes is available
immediately, with no rewrite rule needed to find it; the exact matches this
operator acts on are structural identity, the oldest and cheapest form of
common subexpression elimination, not a search. A chromosome will typically
contain more than one such match, and the operator does not choose among
them: every candidate collapse becomes one additional individual entered
into the population, exactly as existing simplification passes already turn
every candidate simplification into something the tournament scores rather
than something a heuristic selects.

Because a candidate's saving is the number of operations it removes, known
the moment the match is found, every candidate can be ranked before any of
them is evaluated. What the tournament cannot yet do is see this ranking: its
existing dimensions score one expressed output, and the saving from folding
is a property of the whole chromosome — how many genes reference the shared
subtree, and how many operations each reference removes — not a property any
one tree's predictions reveal. A new, chromosome-scoped dimension, one that
reads every gene at once rather than the one gene being folded, is needed
before folding can be judged by the tournament on the same footing as every
other kind of candidate. We have not built this dimension; we record its
absence as the one piece of this proposal that is not yet ready to run.

### Staging and promotion

A winning candidate is staged, not promoted directly: it becomes an entry in
a magic symbol table, a device named after the magic-set transformation used
in query optimization for exactly this purpose — synthesising a new, scoped
object purely so that sharing which would otherwise be invisible to the
system becomes something it can check. An entry records an ordinary symbol
definition in the shape the kingdom's table already uses — a name, a
semantics, a typed arity — together with the concrete subtree it was found
as, literals and variables intact, and where in the population it was found.
Seeding an entry function-field means it is, on its own, no more than a
record of one real, specific saving; it earns no claim to generality by
being staged.

Generality is a separate, later test: take every staged entry, replace its
literals and variables with their types, and group entries by the resulting
shape. A shape with enough independent support behind it — found in more
than one chromosome, not merely repeated within one — is a candidate for
promotion into the kingdom's real symbol table, with its own typed arity,
available from that point to every future individual the search produces.
Whether independent support should be judged by count alone, or should also
require that chromosomes using the candidate actually win tournaments before
it graduates, we leave open; we think the latter is right, because it is the
same standard every other kind of candidate in this system already has to
meet, but we have not tested it.

### Re-encoding

Folding shortens whichever genes contained the now-shared subtree in full,
and the result has to land back inside the fixed gene length every
crossover and mutation operator assumes. The reversible edit this needs
already exists, built for the unrelated purpose of constant folding: build
a single substitution describing what is to replace what and where, hand it
to the existing level-order rebuild, and accept its refusal if the result
cannot close within the gene's fixed length, which leaves the gene
unchanged rather than producing something invalid. What this mechanism does
not yet do is accept more than one substitution referring to one shared
definition in a single pass; whether `FoldToDag` should call it once per
occurrence, independently, or whether the mechanism itself should be
extended to take several simultaneous substitutions against one shared
definition, is an open implementation question rather than a decided one.
The edit itself, in either case, produces a new candidate placed beside the
unmodified original, never a change made in place: nothing in this proposal
is permitted to assert that a fold is correct before the tournament has
judged it.

### Why not a different representation

The case for leaving the genotype alone deserves stating directly, because
it is the first question any of this will draw. Cartesian genetic
programming represents a program as a graph from the start, and library
learning over anti-unified subterms — in the style of DreamCoder, Stitch, or
babble — abstracts recurring structure across a corpus of programs into a
reusable function. Both are legitimate answers to a version of this
problem, and neither is what we propose.

The sharing this document is about is a property of the *expressed*
program, discovered after decoding by equality saturation over the kingdom's
own algebra, not a property of how the genotype is laid out. Nothing about
the karva representation, its fixed-length invariant, or any operator that
acts on it needs to change for this discovery to happen; the one addition is
a reversible edit, built from machinery that already exists, applied after
decoding and before evaluation. A Cartesian representation would have made
sharing visible in the genotype from the start, at the cost of giving up the
guarantee that every offspring of every operator is valid — the single
property that makes gene expression programming the representation worth
using here in the first place.

The relationship to library learning is closer, and worth being precise
about. Anti-unifying across a corpus of already-finished programs, as that
line of work does, is exactly what our promotion step (§4, Staging and
promotion) does to the magic symbol table's entries — the difference is
that our corpus is the population a tournament is actively running over, so
every candidate function arrives already scored by survival, not by a
separate offline corpus analysis. We see this as population-scale library
learning arriving for free from a mechanism built for a narrower purpose,
rather than as a second system bolted on beside the first.

## 5. Evidence and what would overturn it

The first two claims in §2 were checked against two short evolutionary runs
on real symbolic-regression targets, one with a single input variable and
one with four; both ran for the same fixed time, under the same population
and gene-length settings, from the same random seed stream. Full figures,
including the rate of candidate combinations that recurred across
individuals under function-type matching, are in the appendix.

What would overturn the second claim, that structural matching alone is
enough: a kingdom whose own algebra can equate two subtrees that are not
syntactically identical before any folding happens — neither of the two laws
we tested exercised this, since both cases that mattered were structural
identity from the start — would need the full algebraic saturation pass
reinstated rather than skipped as an optimisation.

What would overturn the third claim, that folding is unconditionally a
cheaper way to compute the same chromosome: nothing in principle, since the
edit changes no value a chromosome computes, only how many times it computes
it — but this has not yet been measured on a running tournament with the
chromosome-scoped cost dimension described in §4, only estimated from static
operation counts, and a measured check against wall-clock or device-cycle
cost would be the honest test.

What would overturn the fourth claim, that recurring function-type matches
are worth feeding back into the symbol table: a promoted candidate's
descendants failing to outperform chromosomes without it across repeated
runs, which is precisely the test we propose deferring to the tournament
rather than deciding by count alone.

## Appendix: measurements and implementation pointers

### Measurements

A one-off study (`examples/gene_share_study.rs`) ran both equivalence checks
on two real sixty-second evolutionary runs (eight genes per chromosome, head
length sixty-four, seed 7013): Feynman I.6.2a, one input variable, and
Feynman I.43.43, four input variables, eight hundred individuals in each run.

| | I.6.2a | I.43.43 |
|---|---|---|
| individuals with a function-field repeat within their own genes | 108/800 | 147/800 |
| highest repeat count found in a single individual | 3 | 3 |
| total redundant operator-evaluations saved across the population | 583 | 442 |
| individuals containing a ≥2-operator combination recurring elsewhere in the population (function-type match) | 170/800 | 167/800 |
| most frequent recurring combination | `Sub(ProtectedSqrt(F), F)`, 109 individuals | `ProtectedInv(Neg(F))`, 134 individuals |

A bounded algebraic rewrite pass, run over the same chromosomes after plain
structural matching, found no additional matches on either run.

### Implementation pointers

The karva invariant (fixed head length, tail length derived from it) and the
bidirectional conversion between karva tokens and an expressed tree:
`src/karva.rs`, this repository.

The typed symbol table, its many-hot arity encoding, and a kingdom as a
query over it: `src/geneframe.rs:140` (`Symbol`), `:115` (`Arity`), this
repository.

The existing reversible gene-edit mechanism, built for constant folding: the
edit driver is `Engine::fold_winners` (phylu, `src/evolve/engine.rs:5518`);
the rebuild itself is `vary::relevel` (phylu, `src/evolve/vary.rs:286`),
which refuses and leaves the gene unchanged if the edit will not close
within the fixed length (`vary.rs:329`); the freed token positions after a
successful edit are left as whatever bytes were already there, and remain
subject to ordinary mutation and crossover, which act over a gene's full
fixed width regardless of what currently decodes (`vary.wgsl:539`); the
folded candidate is written beside the unmodified original rather than
replacing it in place (`engine.rs:5753`).

The tournament's scoring mechanism, and its existing support for scoring a
chromosome under a declared subset of its genes: `src/chrom_score.rs`, this
repository.

This document supersedes the treatment of cross-gene reference in
`Geneframe_Homeotic_Genes_design.md`, which proposed a fixed per-chromosome
reference direction with no new symbol table; that design's remaining
correct observations — that the karva round-trip is unaffected by adding a
new kind of terminal, and that the tournament's existing subset scoring
already handles leaving a folded gene out of a chromosome's top-level
combination — still hold and are assumed above.
