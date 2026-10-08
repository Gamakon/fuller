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

Two occurrences `(e, p1)` and `(e, p2)` may share one definition iff
`text(e) = text(e')` and `lineage(e, p1) = lineage(e, p2)`. The definition
is emitted at the latest point `d` that dominates every use, such that
`version(loc, d) = version(loc, p_i)` for every `(loc, _)` in the lineage.
If no such `d` exists (a store to a lineage location sits between the uses,
or the uses lie in different loop iterations), the occurrences are not
shared.

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
| the atomic result, image, call result | its own effect | never shared (as now) |

Whole-array granularity is the first version: `a[i] = x` bumps `a` for
every index. It is sound and it is what the kernels need; distinguishing
provably different constant indices is a refinement with its own test, not
part of this set.

**Loops.** A location stored anywhere inside a loop body (including nested
blocks and callees) takes a *loop version* at the loop header: a value
loaded in the body before the store reads the header version, one loaded
after the store reads the post-store version, and the next iteration's
header version is a new one. In SSA terms the header carries a phi for each
such location. Consequence: an occurrence inside the loop and one outside
it never share a lineage that includes a looped location, and two
occurrences inside the body share only if no store to their locations lies
between them within the body. A `break_if` condition is evaluated at the
end of `continuing`, with that point's versions.

**Branches.** An `if` with a store in one arm bumps the location after the
join (a phi), whichever arm ran. Occurrences in different arms never share
(no dominating point); an occurrence before the `if` and one after share
only if neither arm stores to its lineage.

**Calls.** A call bumps every location its callee (transitively) stores to.
The reader has the callee's roots, so the set is known; a pointer argument
is treated as the whole location it points at.

## 3. Representation

**Reader.** `Node::Leaf`/`App` for a load carries its version:
`load.buffer.xs.#@v12` and `load.local.i@3@v7`, where `v<n>` is the
function-wide version number of that location at the load's program point.
Every node carries `lineage: BTreeSet<(LocationId, u32)>` computed
bottom-up (a leaf's own version plus its children's). A `let` root's uses
carry the root's lineage. Two subtrees are *share-candidates* iff their
rendered text (which now includes versions on loads) is equal; the text
equality is then the lineage equality for free, because versions are in
the text. This is the whole mechanism: the finder stays an exact text
finder.

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
outputs compared bit for bit. A difference is a fault in the representation
or the fold, named by the generated program. This is fuller's own harness
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
- Branches: a store in one arm bumps after the join; arms never share.
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

For adversarial review by phylu-regex before anything is built: the loop
and branch versioning rules (§2) and the placement rule (§1) are the two
places a wrong reading becomes a wrong program again.
