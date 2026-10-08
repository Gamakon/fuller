# PLAN: lineage — memory versions in the WGSL kingdom's DAG

## Context

**Objective.** Make "two subexpressions compute the same value" a property
the graph states, for programs that read and write memory, so that sharing
(the homeotic fold, the DAG extractor, and every mutation that moves a
computation) is legal by construction and not by hand-written guards.

**Approach.** Give every memory location a version: a store makes a new
version, a load reads a named version, and a value's *lineage* is the set
of (location, version) pairs it depends on. Two occurrences of one
expression are the same value if and only if their text is equal AND their
lineage is equal. A shared definition is placed at the earliest statement
where its whole lineage is current. In compiler terms this is memory SSA;
in data-architecture terms it is lineage and dependency modelling; the
e-graph holds it as relations beside its equality classes.

**Unique feature.** The share editor already answers "equal expression"
(exact and after rewriting). This adds "equal history", the half Neumann's
share equivalence needs once operands can change under a plan: the same
subplan over the same relation *versions*. The two together make a fold a
graph property, and the randomised oracle has something precise to check.

**Status.** Design, 2026-10-09. Nothing built. Prompted by the external
review of the step-5 result: "expression equivalence is not execution
equivalence; sharing must respect execution order, memory effects and the
lifetime of computed values."

**Why now.** The step-5 proof found exactly this fault: an inlined `let`
re-read a queue after a store, and the structural gate could not see it
because the reader misread both sides alike. The two rules that fixed it
(a `let` is bound once; a load of a target the function stores is never
shared) are instances of a rule the graph does not yet state. The second
is also wasteful: it refused 40, 36 and 34 repeats in the three largest
kernels, most of them legal.

## 1. The invariant

For an expression occurrence `e` at program point `p`:

```
lineage(e, p) = { (loc, version(loc, p)) : loc read by e }  ∪  lineage of its children
```

where a `let` binding counts as a location whose version is the binding
itself (a use of a `let` reads that binding, never the text re-evaluated).

Two occurrences `(e, p1)` and `(e, p2)` may share one definition iff
`text(e) = text(e')` and `lineage(e, p1) = lineage(e, p2)`. The definition
is emitted at a point `d` that dominates every use, with
`version(loc, d) = version(loc, p_i)` for every `(loc, _)` in the lineage,
and that is in lexical scope of every `let` binding the lineage reads
(WGSL scoping is lexical: a `let` bound in an `if` arm is not visible after
the join, and dominance alone does not say so). Among the legal points,
the one chosen is the EARLIEST outside the deepest loop that contains all
the uses: any legal point is correct, and a point inside a loop recomputes
per iteration. Hoisting an occurrence out of a branch arm to a point before
the branch is legal (WGSL has no traps: indexing is clamped, division is
defined) but runs it unconditionally; that is a cost decision for the
compile-and-time evaluator, not a correctness one, and the oracle cannot
see it. If no legal `d` exists (a store to a lineage location sits between
the uses, a binding is out of scope), the occurrences are not shared.
`version(loc, p)` is well defined only once the phis of §2 are placed:
before that, two paths into `p` can carry different versions and the
number at `p` is whichever path the reader walked last, which is the
reading-error class the step-5 fault belonged to.

What this subsumes: the `let` rule (a `let` is an occurrence with a fixed
lineage at its binding, and its uses carry that lineage, not the lineage of
the text re-evaluated at the use); the conservative load rule (a load of a
stored target is shareable where the version is the same, refused only
where it differs); and the plan's "no intervening store in statement
order", stated for loops too.

## 2. Locations and versions

| location | identity | version bumps when |
|---|---|---|
| a storage or workgroup buffer | the global variable, whole array | any store through it, or a call that may store through it (a callee storing to the same global; a pointer argument) |
| a uniform, a constant, a push constant, an argument by value | the variable | never |
| a local scalar `var` | the local | a store to it |
| a local array `var` | the local, whole array | a store to any element |
| a struct field through a pointer | the base location (whole) | as its base |
| a location reached through a pointer `let` (`let p = &a[i]; *p = x;`) | the base of what it points at (naga resolves it) | a store through the pointer bumps the base |
| a location an atomic acts on | the base | every atomic operation is a store to it (its result is its own effect and is never shared, as now) |
| the atomic result, image, call result | its own effect | never shared (as now) |

**Barriers.** `workgroupBarrier()` and `storageBarrier()` make other
invocations' stores visible: a barrier is a call that stores to every
workgroup location (and every storage location, for the storage barrier),
so two loads with a barrier between them never share. Unsynchronised
stores by other invocations are a WGSL data race and out of scope: the
model is one invocation's program order plus barriers.

Whole-array granularity is the first version: `a[i] = x` bumps `a` for
every index. It is sound and it is what the kernels need; distinguishing
provably different constant indices is a refinement with its own test, not
part of this set.

**Phi placement, the rule.** Versions are placed as memory SSA places
them: at every join of control flow where the incoming versions of a
location differ, the location takes a new version (a phi) at that join.
The joins are computed from the statement tree's dominance frontiers, not
enumerated by construct: `if` (one arm stores, or both arms store different
versions), `switch` (any case stores), a loop header (all its
predecessors: the entry, the fall-through end of `continuing`, every
`continue`), and the loop EXIT (every `break`, `break_if` and the normal
exit, which can carry different versions). Enumerating constructs is how
the `let` fault happened; the frontier rule is the one that holds for
cases not named here.

**Loops, in those terms.** A location stored anywhere in the loop (body,
`continuing`, nested blocks, callees) has a header phi, so a load in the
body before the store reads the header version and one after reads the
post-store version, and an exit phi, so a load after the loop reads
neither of those by number: the same text inside and after the loop are
different versions, as they must be (the post-store version is this
iteration's inside the loop and the last iteration's after it). A
`break_if` condition is evaluated at the end of `continuing`, with that
point's versions.

**Branches.** A phi after the join wherever an arm stores; occurrences in
different arms never share (no dominating point in scope); an occurrence
before the `if` and one after share only if neither arm stores to its
lineage.

**Calls.** A callee's store set is expressed in its own terms (its globals
and its parameters) and substituted per call site: a store through a
`ptr<function>` parameter bumps the caller's argument at that site,
different at every site; a store to a global bumps that global for every
caller. The reader has every callee's roots, so the sets are known.

## 3. Representation

**Reader.** `Node::Leaf`/`App` for a load carries its version:
`load.buffer.xs.#@v12` and `load.local.i@3@v7`, where `v<n>` is the
function-wide version number of that location at the load's program point.
Every node carries `lineage: BTreeSet<(LocationId, u32)>` computed
bottom-up (a leaf's own version plus its children's). A `let` root's uses
carry the root's lineage. A `let` use is rendered with its binding id. Two subtrees are
*share-candidates* iff their rendered text (versions on loads, binding ids
on `let` uses) is equal; with versions phi-placed per §2 the text equality
is the lineage equality, and the finder stays an exact text finder. The
placement check of §1 is then a check on the DEFINITION's point, not a
second test of equality.

**Chromosome.** `exact_shared` drops the conservative load rule; the
version in the text does its job. The `let` rule stays (a `let` is still a
root). `refused_loads` becomes `refused_lineage`: repeats whose texts
differ only in versions, reported so the number says how much sharing the
history forbids.

**Scaffold.** A tail gene gains a *placement*: the statement path before
which it is emitted (the dominating point of §1). The rebuild emits the
definition's expressions there, as one more `let`, and the uses reference
it. Today's tail is emitted at the first use's statement; a definition
whose uses span statements must move to the dominating point or the fold
is refused. The version annotations are stripped at rebuild (the text
`load.buffer.xs.#` is what naga needs).

**Tail evaluation model.** phylu evaluates SR tails once per level before
the heads. For a kernel, a definition inside a loop is recomputed each
iteration by placement, so the kingdom's tail is "definitions placed in
the scaffold", not "definitions evaluated up front". The chromosome records
placement per tail slot; the device decode of the chromosome is unchanged
(placement is scaffold metadata the rebuild consumes).

**E-graph.** Lineage is asserted as relations when the saturated finder
runs over kernel trees (future, when the `arith`/`trig` classes get rules):

```
(relation reads (Math Loc i64))       ; expression e reads loc at version
(relation lineage (Math Loc i64))     ; closure over children
(rule ((reads e l v)) ((lineage e l v)))
(rule ((lineage c l v) (= e (Op c ...))) ((lineage e l v)))
```

and share-legality is "same class AND same lineage set", checked at
extraction. Nothing here changes the Math kingdoms, which have no memory.

## 4. The oracle (the reviewer's test)

A generator of small valid WGSL compute kernels over the kingdom's rows:
one storage buffer in, one out, a uniform, locals, a `for` loop with a
loop-carried local, an `if` with a store in one arm, a nested index, a
`let` before and after a store, a call to a helper that stores. Each
generated kernel is: read, folded (with lineage), rebuilt, and both the
original and the rebuilt kernel are run on the device on random inputs;
outputs compared as bits (`to_bits`, so NaN payloads count), inputs
including NaN and infinities. The generator produces the REFUSAL classes on
purpose, so the gate proves refusals and not only accepted folds: a store
between two textual repeats, a barrier between repeats, a `let` used after
a store to its operand's target, a repeat inside and after a loop, a repeat
in one arm and after the join; and it records, per generated program,
which repeats the fold shared and which it refused. A difference is a
fault in the representation or the fold, named by the generated program.
A thousand kernels with zero differences prove little unless the generator
reaches the programs where folding would be wrong. This is fuller's own harness
(the device run needs no phylu suite): `examples/wgsl_oracle.rs`, under
`gpu,wgsl`, N programs per run, the seed printed, one process.

The same harness is the correctness gate for every mutation that follows.

## 5. What changes, where

| piece | change |
|---|---|
| `src/wgsl/reader.rs` | version table per function; loads named with their version; `lineage` per node; loop/branch/call bumps |
| `src/wgsl/chromosome.rs` | drop the conservative load rule; `refused_lineage`; placement per tail slot |
| `src/wgsl/scaffold.rs` | emit a definition at its placement as a `let`; strip versions at rebuild |
| `src/homeotic.rs` | `FoldedChromosome` gains per-slot placement (an opaque path for the kingdom that uses it); SR unaffected |
| `examples/wgsl_oracle.rs` | the generator and the device comparison |
| `kingdoms/wgsl/README.md` | the measured sharing before/after on the six kernels: repeats found, shared, refused by lineage |

## 6. Tests

- Versions: a store bumps its location and no other; a uniform never bumps;
  a call bumps what the callee stores.
- Loops: a load before and after a store in one body are two versions; the
  same load in two iterations are two versions; a load of an unstored
  location inside the loop shares with one outside.
- Branches: a store in one arm bumps after the join; arms never share;
  `switch` cases likewise.
- Loop exit: a load after the loop is a third version, distinct from the
  header and the post-store versions; a `break` before the store merges
  into the exit phi.
- Barriers and atomics: a workgroup load before and after a barrier are
  two versions; an atomic bumps its location.
- Pointer lets and call sites: a store through `let p = &a[i]` bumps `a`;
  a callee storing through a `ptr<function>` parameter bumps the caller's
  argument at that site only.
- Scope: a definition reading a `let` bound in an arm is not placed after
  the join.
- `let`: a use after a store carries the binding's lineage (the step-5
  case, as a unit test on a four-line kernel).
- Fold placement: two stores sharing a partial whose lineage is current at
  the function start place the definition at the start; the same partial
  with a store between the uses is refused; the same partial inside a loop
  is placed inside the loop.
- The six kernels: round trip and the sweep still hold; the refused counts
  fall and the shared counts rise (numbers in the README).
- The oracle: 1,000 generated kernels, zero differences, seed recorded.

## 7. Order

1. Versions and lineage in the reader, with the unit tests (no device).
2. The oracle harness over the six kernels and 100 generated programs
   (one device process, announced).
3. Fold placement and the chromosome change; six-kernel numbers.
4. Then, and only then, the first mutations: the `Math` rewrite rules on
   `arith`/`trig` regions with no loads, gated by the oracle.
5. The compile-and-time evaluator on top.

## Review

Reviewed adversarially by phylu-regex (2026-10-09, twelve findings, all
folded in): the loop exit phi, phi placement by dominance frontiers
instead of by construct, barriers and atomics as stores, `let` scope and
binding order in placement, the placement point stated once (earliest
legal outside the deepest loop), hoisting out of an arm as a cost
decision, pointer lets, per-call-site substitution, all loop predecessors,
the iteration sentence dropped, versions well defined only after phis, and
the oracle generating the refusal classes with bit comparison and NaN
inputs.
