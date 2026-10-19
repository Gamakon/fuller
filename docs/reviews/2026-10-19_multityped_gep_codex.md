# Adversarial review: Multi-typed GEP

Verdict: the per-type tail idea can give type closure and its tail bound is
sound, but the specified operators and representation do not yet do so in all
cases.  It is also no longer a Karva reading of one ordered head when there is
more than one type.

Notation in examples: `f:R(R,A)` returns `R` and takes an `R` then an `A`;
`r:R` and `a:A` are terminals.  Head positions are left to right.

## 1. Closure under every section-5 operator: no

**Dc-to-head transposition is a counterexample.**  Let `C_I=[1,2]`, with a
Dc-I entry `1`, and let the head alphabet be symbol IDs (`f`, `r`, `?_I`,
...).  Section 5 permits an IS/RIS sequence “from any domain” into the head.
Moving `[1]` from Dc-I into a head position leaves a constant-array index in a
domain that must contain a symbol.  It has no output type, so expression
cannot select a type-correct node without rejecting or interpreting it as a
substitute.  **Fix:** IS/RIS sources must be head or a typed tail only (and
preserve a fixed head length); Dc recombines/transposes only with the aligned
Dc domain.  RIS additionally needs the usual “source starts at a function”
rule.

For the remaining stated domain-preserving operations, closure follows if
each table row really has one output type and every used function has an
ordered, checked input signature.  The latter is missing.  A many-hot vector
does not say whether `f:R(I,S)` is called as `(I,S)` or `(S,I)`.  **Fix:** add
an ordered argument/signature relation, e.g. `(symbol, arg_index,
required_type)`, and use it to append slots.  Counts can remain a derived
index.  Otherwise a mixed-input emitted call can be type-wrong despite every
node satisfying a multiset of type demands.

## 2. Tail bound: yes, subject to the preceding invariants

`D_T <= [T=R] + h*a_T`: the root creates one `R` demand and each of at most
`h` expressed head functions creates at most `a_T` `T` demands.  A tail-T
entry is consumed only by a `T` demand, hence it cannot need more than this.
Also every tail entry is a leaf, and with at most `h` functions and maximum
arity `n_max`, total leaves are at most `h*(n_max-1)+1`.  Taking the minimum
of these two upper bounds is therefore safe.

The attempted edge cases do not break it.  A root terminal consumes one head
row (or one R-tail row if there is no R head row) and opens no slots; a head
terminal only reduces later demand; a polymorphic spelling represented by
separate, one-output rows is harmless; and a mixed-input function contributes
its separate counts to the corresponding bounds.  For one type,
`a_T=n_max`, so the minimum is Ferreira's exact `h*(n_max-1)+1`.

**Needed condition/example:** `f:R(I,S)` cannot be stored merely as
`in_I=1,in_S=1`; the decoder needs `[I,S]` (or a canonical argument ordering
defined by the kingdom).  **Fix:** make this ordered signature part of the
kingdom validity check.  Also define `n_max` over the same function rows and
forbid zero-output / multi-output rows.

## 3. Frames: deterministic, and single-type equality is correct

Given a total head order, a unique output type per row, tail orders, and an
ordered signature, `(slot type, k_T)` selects exactly one row.  With one type,
`H_T` is the whole head and the frame is head followed by tail: exactly
classical Karva decoding.

The published data model does not yet provide all of that determinism.  It
has only type *counts*, not argument order, and no gene/domain-local sequence
column for multiple tail-T domains.  Example: the two equally legal
interpretations of a row with `in_I=1,in_S=1` yield different ordered calls.
**Fix:** add `gene_id`, `domain`, `domain_type`, and `domain_seq_id` (or an
equivalent unique ordering key), plus the ordered signature relation.

## 4. It loses a central Karva property

It retains neutral material--for example head `[r:A, f:R(R,A), r:R]` with
root `R` leaves the leading `A` row unexpressed--but it is not one Karva
level-order string.  More sharply, with head
`[f:R(R,A), a:A, r:R]`, root `R`, expression visits the child slots `R,A` and
reads head positions `1,3,2`, producing `f(r,a)`.  Thus it skips and later
returns to an earlier head row; expressed head order is not chromosome order.
Classical Karva always consumes an ordered prefix of its head before its tail.

This also weakens classical GEP's positional locality: mutate that `a:A` to
an `R` terminal and it leaves the A frame and enters the R frame, shifting
all later uses in both frames.  **Fix:** either accept and describe this as a
new typed, interleaved-frame representation rather than Karva/GEP proper, or
redesign around one globally consumed head stream with a construction that
guarantees the demanded type.  The latter cannot coexist with unconstrained
arbitrary-type head mutation without another mechanism.

## 5. Section-3 SQL: no

It is a sketch, not correct executable SQL against the supplied model.
`type` is ambiguous/NULL for head rows under section 7; it must use the
joined symbol's output type for heads and the domain type for tails.
`is_tail` does not exist (section 7 supplies `domain`), `symbol_types` is not
defined, and `symbols` and `terminals` require a UNION to form that relation.
`PARTITION BY type` leaks across genes/chromosomes, while `gene_seq_id` alone
cannot order separate tail-T domains.  Finally this frame query does not
perform the recursive level-order slot walk or validate function signatures.

Example: two genes in one chromosome, each with an `I` tail whose local
position is 0, are merged into one I frame.  **Fix:** materialise a unified
one-output `symbol_outputs` view; partition/order by chromosome, gene,
domain type and local position; construct head output type from that view;
and implement a recursive CTE/UDF queue keyed by `(type,k_T)` and ordered
argument slots.  Do not call the frame ranking itself the walk.

## 6. Missing multigenic, homeotic, and Dc details

**Multigenic.** “Each gene has” does not specify gene boundaries, equal versus
per-gene layouts, a linking function/phenotype root, or whether crossover is
at homologous gene and domain coordinates.  Example: two `R` genes need an
explicit `link:R(R,R)` (or a fixed external combiner); merely evaluating both
does not define the chromosome's result.  **Fix:** add `gene_id`, a chromosome
linker with ordered typed inputs, and restrict gene recombination to matching
gene/layout IDs.

**Homeotic.** A reference terminal needs a declared output/root type *and*
the referenced definition must be immutable, in scope, and acyclic.
Example: `H1:R = f(H2:A,...)` and `H2:A = g(H1:R)` type-check locally but is
non-terminating.  **Fix:** store root type and stable ID on each homeotic
definition, reject/self-prevent cycles (or require lower levels only), and
state the lifecycle/visibility rule during mutation and recombination.

**Dc.** The per-type Dc idea is compatible with GEP-RNC, but it needs an
unambiguous traversal counter and domain-local ordering.  Example: head
`[?_I, ?_I]` and Dc-I `[4,9]` should use `4` for the sole expressed root and
leave the second placeholder and `9` neutral; a global symbol occurrence
rank would instead be wrong when other types interleave.  **Fix:** define a
separate `q_T` incremented only when an *expressed* `?_T` is visited, index
`Dc_T[q_T]`, make Dc length `t_T`, and constrain mutation/recombination to
the matching `(gene,type,Dc)` domain.  Specify whether constants are
per-gene or shared and how constant-array evolution preserves referenced
indices.
