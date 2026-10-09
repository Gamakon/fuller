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

**Status.** Design 2026-10-09, approved by four reviews; steps 1–3 of §7
BUILT and measured the same day (`kingdoms/wgsl/README.md`): memory
versions in the reader (afae87a), `legality::decide` with placement per
tail slot (f436a90), the reference interpreter (36e4860), the oracle's
hand-written set (4c0827d) and generator (31e497e, 1,000 of 1,000 agree
three ways on Apple M3 Max; three platform findings recorded); step 4,
placement in the rebuild (`rebuild_folded`), built and proven on the
device with the same 1,000 programs (a fourth finding: sharing a float
partial changes Metal's contraction by an ulp or so, so the evaluator
needs the numerical criterion from its first float kernel); step 5, the
first mutations (`mutate.rs`, the Algebra family over arith/trig regions
with opaque leaves), 426 of 426 mutants of 200 generated programs pass
the criterion on the device, and the six real kernels hold only ten such
regions (score, hff): the engine's kernels are integer code. Step 6 (the
evaluator) not started. Prompted by the external review of the step-5 result:
"expression equivalence is not execution equivalence; sharing must
respect execution order, memory effects and the lifetime of computed
values."

**Why now.** The step-5 proof found exactly this fault: an inlined `let`
re-read a queue after a store, and the structural gate could not see it
because the reader misread both sides alike. The two rules that fixed it
(a `let` is bound once; a load of a target the function stores is never
shared) are instances of a rule the graph does not yet state. The second
is also wasteful: it refused 40, 36 and 34 repeats in the three largest
kernels, most of them legal.

## 1. The invariant

Stated once, as the external reviewer put it: **sharing requires
equivalent values, compatible memory histories, valid dominance and safe
evaluation placement.** The four are decided together, as one property of
the annotated DAG (§5, `legality`), never as separate checks spread over
the reader, the chromosome and the scaffold.

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

**Safe evaluation at `d`.** WGSL expressions are total: indexing is
clamped, integer division by zero and float overflow are defined, there
are no traps and no exceptions, so evaluating `e` at `d` with the same
lineage yields the same value it would have inside its original branch,
and the branch's guard changes only whether the value is USED. Safety is
therefore implied by lineage, dominance and scope in this kingdom. It is
still a named condition of the invariant, because a kingdom whose
expressions can trap (a CPU language, a kingdom with asserts) would need
it decided, and because two WGSL cases are excluded explicitly: an
expression containing a derivative or a barrier-dependent builtin is never
moved (uniform control flow), and an expression whose evaluation is
unbounded in time does not exist in this language.

**What may be shared at all (pure and deterministic).** Equal text and
equal lineage imply equal values only for expressions that are functions
of their operands and of the invocation. The admissible set is stated, not
assumed: rows of the classes `arith`, `trig`, `geom`, `index`, `count`,
`int`, `fixed`, `convert`, `quantise`, `bits`, `hash`, `compare`, `logic`,
`select`, `shape`; literals; loads, under their version; `let` uses, under
their binding; and the invocation builtins (`global_invocation_id`,
`local_invocation_id`, `local_invocation_index`, `workgroup_id`,
`num_workgroups`), which are constant within one invocation's program and
so shareable within it. Excluded from sharing, always: a call result (its
own effect unless the callee is proven to store nothing and read only
versioned locations, which is a later refinement, not assumed), atomic
results, image and sampler operations, derivatives, anything reading a
barrier-visible location across a barrier, and any builtin not in the list
above. Floating-point operations are deterministic for a given device and
compiler; the oracle compares two programs compiled by the same compiler on
the same device, so contraction (`fma`) does not enter until the
algebraic rewrites of §7 step 4, which get their own criterion.

**Iterations are explicit.** A location's header phi is a version distinct
from the pre-loop version, so "inside the loop" and "before the loop" are
different lineages by construction, and a definition may be placed OUTSIDE
a loop only when no location in its lineage has a phi at that loop: that
is the proof of loop invariance, and nothing is hoisted without it. Within
one iteration two occurrences with the same static version read the same
value because the shared definition is itself placed inside the loop and
recomputed each iteration; the static version names the loop-carried value
per iteration, never one across iterations.
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

**E-graph: effect-qualified identities, not lineage checked later.** A
memory-dependent expression never enters an unrestricted equivalence
class: a load is the term `(Load loc version)` with its version as part of
the term, a `let` use is `(Bind id)`, so two loads of different versions
are different terms and no rewrite can merge them; once an unsound equality
has merged two classes, checking lineage at extraction is too late. With
that, lineage is a function of the term and needs no side relation:

```
(datatype Expr ... (Load Loc i64) (Bind i64) ...)
```

Rewrite rules apply to pure subgraphs (`arith`, `trig`, …) whose leaves
may be effect-qualified terms; a rule never mentions `Load` or `Bind`. The
Math kingdoms have no memory and are unchanged.

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
reaches the programs where folding would be wrong. Beside the generator, a
fixed set of adversarial kernels written by hand and kept in the repo:
aliasing through pointers, conditional stores inside loops, dynamic
indexing with repeated loads, workgroup memory with barriers, expressions
that are valid only within a branch (an index guarded by `if (i < n)`),
and the step-5 queue case.

Bit equality is the criterion while expressions are unchanged. The
arithmetic rewrites that come after (§7 step 5) reorder floating-point
operations and need their own numerical criterion, decided here before
the first such rewrite is admitted, never by loosening this gate.

**The numerical criterion (decided 2026-10-09, from the measurements).**
A mutant is compared with the ORIGINAL on the device, same compiler, on
the oracle's inputs, never against the interpreter. It is accepted when:
every integer lane is bit-exact; every lane that is NaN or infinite in
the original is the same class in the mutant (NaN against NaN, an
infinity of the same sign), and no lane is non-finite on one side only;
and every finite float lane is within 64 units in the last place of the
original. Where 64 comes from: the platform leaves Metal's fast math on,
and the step-4 run measured what that alone does to an unchanged
expression, sharing moved 34 lanes of 24,000 by up to 8 ulps and a
longer chain by up to 64 (the interpreter's "far" class), so 64 ulps is
the noise floor of expression-preserving changes on this device, and a
rewrite that moves a lane further has changed the value, not the
rounding. Signed zeros are equal. The bound is per lane, not aggregate:
one lane past it refuses the mutant.

**Tiering (phylu's advisory, 2026-10-09).** The per-lane criterion is the
oracle's smoke test: it catches a wrong program, it does not certify a
kernel for the engine. phylu reads a score by its RANK, never as a
number: HFF selection, the tournament, the pump, cohort promotion and
the −19 / 1e-10 gate are comparisons between rows, a one-ulp move on a
near-tie flips a survivor and the population diverges within a
generation, and the engine's contract is bit-identical determinism. So:
decode, cand, vary, evolve (discrete outputs) are gated bit-exact,
always; score and hff are gated SELECTION-IDENTICAL on real populations,
which is the override sweep (the pinned golden through real generations,
the sample gate, exact resume), with the lane test as the smoke test
only. Three amplifiers make lane ulps the wrong unit there: reductions
(a score sums over rows), near-zero outputs (1−R² at a law is ~1e-11,
where the gate's absolute threshold sits), and the OLS wrapper's 2×2
solve on an ill-conditioned row. A float share or rewrite in score/hff
that is legal by lineage may still be refused by the golden; that is a
measurement of the engine's sensitivity, not a flaw in the gate. The
compile-and-time evaluator measures at the decision level for those two. This is fuller's own harness
(the device run needs no phylu suite): `examples/wgsl_oracle.rs`, under
`gpu,wgsl`, N programs per run, the seed printed, one process.

The same harness is the correctness gate for every mutation that follows.

**A third implementation.** The reader and the rebuild could share one
wrong assumption, as they did in the `let` fault; the oracle compares the
original and the rebuilt program through the SAME compiler and device. So
a small reference interpreter, written independently of both
(`src/wgsl/interp.rs`: scalar `f32`/`i32`/`u32`/`bool` rows, loads and
stores on host arrays, the scaffold's loops and branches, one invocation at
a time), executes the chromosome directly on the oracle's inputs, and its
outputs are compared with the device's. Three implementations, any two
agreeing against the third point at the faulty one, but a majority is not
an authority: the ORIGINAL kernel's device output is the reference, and
for the hand-written adversarial kernels the expected outputs are written
down by hand with the kernel (computed independently, never copied from a
run). When the interpreter and the device disagree on the original, that
is a finding about the interpreter or about naga, settled against the
hand-written expectation, not by vote. Vector and matrix rows join the
interpreter when a kernel in the set needs them.

## 5. What changes, where

The legality decision is one function on the annotated DAG, in one place:

```
legality::decide(dag, candidate_sites) -> Accept { placement } | Refuse { reason }
```

where `reason` is one of: not admissible (§1's pure set), text differs,
lineage differs, no dominating point, operand not available at the
placement (a `let` binding or a local not yet defined there, or out of
lexical scope), version not current at the placement (a bump between the
placement and a use), unsafe to move (derivative/barrier builtin), not
loop-invariant for the placement asked.

**The placement invariant, tested as such.** Version equality between the
sites does not by itself make a placement legal. `decide` establishes, and
a test pins for each clause: every operand of the definition is defined
before `d` in evaluation order and in lexical scope at `d`; `d` dominates
every use; for every lineage location the version at `d` equals the
version at every use, with no bump on any path from `d` to a use. The
tests construct a kernel per clause where exactly that clause fails. The fold, the rebuild, the
report and every later mutation call this and nothing else; a mutation
that moves or merges a computation is legal iff `decide` accepts it. That
is the reviewer's "first-class property of the DAG", and it is what makes
the optimiser reasoned about in one place.

| piece | change |
|---|---|
| `src/wgsl/legality.rs` (new) | the decision above, over the lineage-annotated DAG; the reasons are counted in the report; the placement invariant's clause tests |
| `src/wgsl/interp.rs` (new) | the independent reference interpreter of the chromosome, scalar rows first |
| `src/wgsl/reader.rs` | version table per function; loads named with their version; `lineage` per node; loop/branch/call bumps |
| `src/wgsl/chromosome.rs` | drop the conservative load rule; the fold asks `legality::decide`; refusals by reason; placement per tail slot |
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
- The admitted rows at WGSL's edges, each a hand-written kernel with its
  expected output written beside it: float division by zero, `sqrt` and
  `log` of negatives, overflow to infinity, NaN through `min`/`max`/
  `select`/comparisons, `-0.0`; integer division and remainder by zero
  and `i32::MIN / -1` (WGSL defines both), shifts by 32 or more, wrapping
  add and multiply; out-of-range indexing (clamped by naga's bounds
  checks, so the rebuilt kernel must keep the same policy), negative
  `i32` indices, `u32` to `i32` conversions and `f32` to integer
  conversions out of range. The interpreter must reproduce the WGSL
  definition, not Rust's, for every one of these.

## 7. Order

1. Versions and lineage in the reader, with the unit tests (no device).
2. `legality::decide` with the clause tests; the reference interpreter
   over the scalar rows, checked against the six kernels' stored answers
   where phylu's suite gives them (no device).
3. The oracle harness: original on device, rebuilt on device, chromosome
   in the interpreter, over the adversarial set and 100 generated
   programs (one device process, announced).
4. Fold placement and the chromosome change; six-kernel numbers.
5. Then, and only then, the first mutations: the `Math` rewrite rules on
   `arith`/`trig` regions with no loads, gated by the oracle.
6. The compile-and-time evaluator on top.

## Review

External review, fourth pass (2026-10-09): approved for implementation,
with one qualification, folded in: the original kernel and hand-written
expected outputs are the authority when the three implementations
disagree (§4), and the admitted rows get edge-case tests for
floating-point behaviour and indexing (§6). The design is not to be
expanded further before implementation evidence.

External review, third pass (2026-10-09), folded in: the admissible
(pure, deterministic) set stated before sharing; the placement invariant
stated and tested clause by clause (availability, scope, dominance,
version currency on every path); an independent reference interpreter as
the third implementation in the oracle. Verdict: proceed with the reader,
memory SSA and legality first.

External review (2026-10-09, after the first revision), folded in: the
invariant restated as four conditions decided together; safe evaluation
at the placement point made explicit (total in WGSL, excluded builtins
named); iteration semantics made explicit (header phi distinct from the
pre-loop version; hoisting only on proven invariance); effect-qualified
terms in the e-graph instead of lineage checked at extraction; the
adversarial kernel set; a separate numerical criterion for the later
floating-point rewrites; legality as one first-class decision on the DAG.

Reviewed adversarially by phylu-regex (2026-10-09, twelve findings, all
folded in): the loop exit phi, phi placement by dominance frontiers
instead of by construct, barriers and atomics as stores, `let` scope and
binding order in placement, the placement point stated once (earliest
legal outside the deepest loop), hoisting out of an arm as a cost
decision, pointer lets, per-call-site substitution, all loop predecessors,
the iteration sentence dropped, versions well defined only after phis, and
the oracle generating the refusal classes with bit comparison and NaN
inputs.
