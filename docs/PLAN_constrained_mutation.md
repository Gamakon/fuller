# Constrained generating mutation for WGSL kernels

Design, 2026-10-12. Status: **revision 2, after the Codex adversarial review**
(`logs/HGP/001/codex_review/plan_constrained_mutation_review.md`; its six findings
and what changed are in section 12). Next: Andrew's review. Nothing is built yet.

## Context

**Objective.** Make evolution of WGSL compute kernels work without language
models: a mutation and crossover that produce, every time they fire, a
program the GPU will compile and run legally, and that change it in the ways
GPU kernels are actually tuned, so that selection can climb.

**Why now.** On 2026-10-11 the evolution pipeline became fast and stable
(about 0.4 to 0.8 s a fitness evaluation, every generation fully feasible
through the reserve refill, timing noise caught), but the variation operators
did not climb. In the last run 1,040 evaluations over 50 generations found
nothing better than the starting kernels (best HFF 0.0964, about 10.5 times
faster than the shipped kernel, set in generation 0). Measured operator
quality the same day:

| operator | measure | result |
|---|---|---|
| engine head mutation (one symbol) | feasible on first evaluation | 16 of 92 |
| typed subtree crossover | correct children on the device | 10 of 40 (1 of 40 positional) |
| any native operator, 50 generations | improvement of the best | none |

A one-symbol change in a kernel is usually a catastrophic change of meaning:
a block factor appears in the grid header, the loop bounds, every stride and
the store guard, and changing one copy writes the wrong cells. Kernels are
tuned by a small set of coordinated changes (tile, block, unroll, vectorise,
band), each touching several places at once.

**Approach.** Mutation becomes a rule system in the sense of Holland's
constrained generating procedures, of GEP, and of `sed`/`awk`: a rule is a
pattern with typed holes, a replacement, and constraints; a rule fires only
where its pattern matches and its constraints hold, and its output is legal by
construction. Three constraint mechanisms, as Andrew named them: joins
between tables, pattern matching, and lookups in the magic symbol table.
Rules come from three sources: the language's structure, the catalogue of
known kernel transformations, and rules mined from the differences between
parent kernels and the faster kernels the models wrote.

**Unique feature.** The gene is a full WGSL syntax tree whose constructs are
blocks (a loop is one C-shaped symbol with a typed slot for its body, as in
Water Bear), coupled values are tied by a mutation-group tag in the symbol
table, and every mutation is a rewrite on that tree, so legality is a
property of the rule set, not of luck.

**What this extends, not replaces.** Andrew's own designs:
- the **magic symbol table** (`docs/Evolving_DAGs_with_GEP_v3.md`, after the
  magic-set transformation of query optimisation): staged subtrees, generalised
  by replacing literals and variables with their types, promoted when they win;
- the **replacement dictionary** (memory, 2026-10-01): pattern -> replacement
  pairs keyed in fuller's canonical form, fired as mutations, persisted and
  pruned by evidence across runs;
- **typed snap** (2026-09-27): a subtree near a library shape is switched to it;
- the **functional-wrap mutation rule** (2026-09-22): mutate with coherent,
  meaningful edits on the edge cases, not general tree edits;
- the **share-equivalence fold** and homeotic tail (`src/homeotic.rs`,
  `docs/PLAN_population_dag.md`): one shared definition referenced by many uses.

## 1. What "legal" means

A program is legal when it parses, validates under naga, compiles on the
device, terminates, writes only the output cells its thread owns, and reaches
every barrier in uniform control flow. Legal does **not** mean correct values.
Andrew's rule (2026-10-11) stands: a legal program that computes wrong values
gets the worst fitness and dies in selection. Rules are judged by the share of
their children that are legal (target: all) and by how often their children
are faster (the evolutionary signal); correctness is the fitness's job.

**Legality is checked, not assumed (review finding 1).** Joins, patterns and
table lookups choose where a rule may fire; they are not a proof. After every
rewrite the child passes a layered checker, and a child that fails it is a bug
in the rule (counted, the rule's record lowered), never sent to the GPU:

1. typed tree and lexical scope (every symbol read is in scope, every slot typed);
2. the structural safety grammar kept from today's kingdom: a store to the
   output only through the owned-cell type, barriers only in non-conditional
   statement positions, loops only from the bounded loop constructors
   (`functions.tsv` store and loop rows, `loader.rs` barrier duals);
3. range proofs: every index in its buffer's or array's range, the binding
   range inside the buffer, every loop bound uniform or capped at the declared
   maximum trip count. `u32` overflow is NOT illegal (decision 2026-10-12):
   WGSL defines it as wrapping, and with every access masked or owned and
   every loop bounded, a wrapped value can only give wrong values, which
   selection removes; the checker reports it as a diagnostic;
4. barrier uniformity by control dependence, not by position alone;
5. adapter checks: workgroup size, workgroup memory, subgroup and other
   features against the selected adapter's limits and features (today
   `render.rs` validates with every naga capability enabled; that is a bug
   to fix first, so validation matches the device);
6. bounded loops: every loop's trip count bounded by a literal or a uniform.

Mined rules (section 6) are untrusted: they become rules only after the
checker passes their children across the corpus, and they stay subject to it.

## 2. The gene: a full WGSL tree of blocks

### 2.1 Coverage

The kingdom today covers a narrow, Gram-specific part of the language
(bindings named `h`, `phi`, `cfg`; four fixed workgroup arrays; workgroup
dimensions from {4, 8, 16, 32} per axis). The target is the whole language
for compute shaders: every statement, expression, type (scalars, vectors of
every width, matrices, arrays including runtime-sized ones, structs, atomics,
pointers), address space, attribute, compute built-in value and built-in
function with its overloads. The coverage table is being built
(`docs/kingdom/WGSL_COVERAGE.md`, kingdom work in progress 2026-10-12), with
the tuning constructs first: 1-D and any legal workgroup shape, workgroup
arrays sized by the gene within the 32 KB budget, full unrolling, tree
reductions, integer atomics, subgroups where the adapter supports them, and a
coalescing layout choice.

### 2.2 Blocks

A construct whose parts must agree is one symbol. Its arity is its typed slots.

| block | slots | what it ties together |
|---|---|---|
| `for` loop | counter, bound, step, body (a statement sequence) | header and end, the counter's scope |
| `if` / `if-else` | condition, then-body, else-body | both branches' scope |
| tile phase | load body, barrier, compute body, barrier | the barrier pair around a shared tile |
| guarded store | cell index, guard, value | a store and the guard that keeps it in range |
| tree reduction | workgroup array, length, combine | stride-halving loop, barriers, the final read |
| band | axis, band size, partial body, combine | a split reduction and its combine (section 5.3) |
| kernel | interface, grid, body, epilogue | the dispatch contract and the store (exists: the store epilogue root of fuller af092b3) |

In GEP terms the head picks blocks and the tail fills their slots; a block's
typed slots accept only symbols of the slot's type, as today's duals do.

### 2.3 The mutation-group column

Beside arity, the symbol table gets a **group** column. Symbols in the same
group are coupled: a mutation that changes one member changes every member
of the same group in the individual, in step. Examples:

| group | members |
|---|---|
| block factor (x) | the grid header's bx, the register block's x length, the epilogue's column loop bound, the column stride, the store guard's column offset |
| tile length | the workgroup array's length, the load loop's bound, the compute loop's bound, the index mask |
| workgroup shape | `@workgroup_size(x, y)`, the dispatch rule, every `lid` stride |

**Implementation: first-class parameter nodes (revised after review finding 2).**
A group is a typed **parameter** of the kernel tree: a named node under the
kernel root (for example `param.block_x : blk`) with its value, and every
dependent position is an expression of that parameter (`block_x`,
`block_x - 1u` as a mask, `wx * block_x` as a stride). A knob mutation
changes the parameter's value; the renderer writes every dependent position
from it, so one change is coherent by construction. Parameters are closed:
they never mention loop counters, registers, bindings or locals, so they are
context-free and survive crossover (a grafted block that reads `block_x`
reads the receiving kernel's parameter; the graft is allowed only when the
receiver declares it).

Parameters are **not** the deduplication `href` mechanism: today's tail slots
are whole, independently Karva-encoded genes addressed by position
(`safe_karva.rs`, `homeotic.rs`), which is exactly what broke positional
crossover. Lowering parameters to shared rows comes later, only through the
stable-row-id population design (`docs/PLAN_population_dag.md`) with
dependency closure, remapping and size budgets. The reader assigns
parameters when it reads a kernel (its ownership proof already identifies the
block factor, the workgroup shape and the tile lengths).

### 2.4 The interface (generic bindings)

The kernel root's first slot becomes a declared **interface**: the bindings
(name, address space, element type, access mode), the uniform struct's fields,
and which buffer is the output and how a thread's cell is named. The Gram
kernel's `h`/`phi`/`cfg` become one interface among many. The ownership proof
and the guarded-store block read the interface instead of fixed names. This
is what makes the kingdom serve other problems (UltraDim's trellis beam,
query projection, the exact re-score; `docs/WGPU_EVOLUTION_TARGETS.md`).

### 2.5 The dispatch plan

Some tuning lives on the host: how many dispatches, which buffer ranges each
binds, padding allocations, the order of passes. The gene's root gains a
**dispatch plan** slot: a sequence of passes, each a kernel with its grid,
its bound ranges and its band. This is needed for padding (section 5.2),
banding across dispatches (section 5.3), the buffer limits that force
banding (Metal's 128 MiB per binding; the driver's `effective_buffer_limit`),
and autotuning (section 7).

## 3. The rule

A rule is

```
rule  name
  find      a pattern over the tree, with typed holes  ?x : type
  where     constraints over the bound holes
  replace   a tree over the same holes and fresh symbols
  groups    the mutation groups the replacement changes in step
  legal     the proof obligation the rule discharges
```

Firing a rule:

1. **Match.** Find every site where `find` matches the individual's tree
   (fuller's e-graph matcher, section 4.2). This is the set of rules that
   "light up" for this individual, with their sites.
2. **Choose.** Pick one (rule, site) by the operator's policy (uniform, or
   weighted by the rule's record of winning, section 6.4).
3. **Bind and constrain.** Bind the holes; evaluate the `where` constraints
   (joins, lookups, pattern facts; section 4). A rule whose constraints fail
   at a site does not fire there.
4. **Rewrite.** Build the replacement, apply its group changes, re-encode the
   tree into the gene at the population's layout (the path model kernels
   already take: `wgsl_model::read_back`).

### 3.1 Worked example: block factor 4 -> 8 on the columns

```
rule  block-x
  find      kernel(?iface, grid(?wx, ?wy, ?bx, ?by), ?body, ?epi)
  where     ?bx' in magic.blk                      -- lookup: {1, 2, 4, 8}
            ?bx' != ?bx
            regs(?bx' * ?by) <= budget.registers    -- join: register table
            ?wx * ?bx' divides tile.cols(?body)      -- join: the tile's column length
  replace   kernel(?iface, grid(?wx, ?wy, ?bx', ?by), ?body, ?epi)
  groups    block-factor-x := ?bx'
  legal     the store epilogue's column loop runs 0..?bx' with the guarded
            store; every column index is wid.x*?wx*?bx' + lid.x*?bx' + k,
            k < ?bx', proved by the ownership proof from the counter bound
```

Because `block-factor-x` is one shared definition, the replacement changes
the grid header, the register block's column length, the epilogue's column
loop bound and the column stride together. The register-budget join forbids
an 8 x 8 block where the registers run out. The tile join keeps the
workgroup's columns a divisor of the tile's columns, so the load loop still
covers the tile.

## 4. The three constraint mechanisms

### 4.1 Joins between tables

Constraints that relate one part of the program to another are joins over
the kingdom's tables and the individual's facts:

- `types.tsv` / `functions.tsv`: which rows accept which duals (a slot's type
  joins the candidate symbol's output type; this is today's typed projection);
- the **facts table** of an individual, extracted by the reader: for each
  group its value, each loop's bound, each workgroup array's length and type,
  registers in use, threads per workgroup;
- the **device table**: the adapter's limits (invocations per workgroup, each
  axis's maximum, workgroup memory, binding size, subgroup support).

`?wx * ?bx' divides tile.cols` is a join of the grid fact with the tile fact.
Joins are cheap (a few rows per individual) and evaluated on the host.

### 4.2 Pattern matching

`find` patterns are matched with fuller's e-graph (egglog). fuller already
runs WGSL regions through an e-graph (`src/wgsl/mutate.rs`, arithmetic
regions with opaque leaves; `src/ruleset/share.rs`). **Revised after review finding 4.** An e-graph's equivalence unions are not
an ordered-effect representation: putting loads and stores into sequence
terms does not stop a class from equating terms with different effects. So
the work splits:

- **statement and block rules** match on a typed control-flow tree with
  explicit sequencing, effects (which buffer and range each statement reads
  and writes) and barrier regions, by a conventional typed matcher (a tree
  pattern with typed holes, matched by walking the tree);
- **egglog stays on pure, isolated expressions** (index arithmetic, value
  arithmetic, as `mutate.rs` does today), where equivalent spellings merge
  (`(wid*16 + lid)*4` and `wid*64 + lid*4`, the case the reader had to
  normalise by hand on 2026-10-11), in small rule families with hard node and
  time caps (the crate's non-confluence rule, `CLAUDE.md`).

### 4.3 Lookups in the magic symbol table

Andrew's magic symbol table holds staged patterns: subtrees found in the
population with literals intact, generalised by replacing literals and
variables with their types, promoted when they win. Two uses here:

- **literal sets** (today `literal_sets()`: block factors, workgroup
  dimensions, unroll factors, index literals): a knob's new value is a lookup
  in its set;
- **mined rules** (section 6): a staged pattern -> replacement pair is a rule
  whose `find` and `replace` come from the table; firing it is a lookup of the
  individual's subtrees against the table's patterns.

## 5. The rule catalogue

### 5.1 Knob mutations (change performance parameters coherently)

| knob | values (lookup) | constraints (joins) |
|---|---|---|
| workgroup shape, incl. 1-D | x*y <= max invocations; each axis <= its max | dispatch rule; strides |
| block factor x / y | {1, 2, 4, 8} (and the set grows by evidence) | registers; tile divisibility |
| tile length / array length | lengths within 32 KB total | load loop covers the tile |
| unroll depth | {1, 2, 4, 8, full} | trip count literal for full |
| vector width | {1, 2, 4} | alignment of the row length or a guarded tail |
| coalescing axis | x or y fastest | the load's index layout |
| reduction strategy | per-thread loop, tree, two-pass, atomic (integer only) | section 5.3 |

### 5.2 Meaning-preserving rewrites (always correct, different speed)

From the standard catalogue of loop and memory transformations. **Revised after
review finding 3:** every precondition is stated over an explicit model of
effects (buffer and range read and written per statement), aliasing (which
bindings may overlap), atomics, barrier phases and control dependence; each
rule ships with counterexample tests. Reductions state a numerical tolerance
as their contract, not exact equality.

| rewrite | precondition |
|---|---|
| tile / strip-mine a loop | trip count known or guarded tail |
| interchange two loops | no loop-carried, alias, atomic or control dependence across the pair, and no barrier inside either loop |
| unroll (partial, full) | literal trip count for full |
| fuse / split loops | as interchange, and barrier phases unchanged (no fusion of two barrier loops) |
| hoist a loop-invariant load or expression | the VALUE is invariant: no barrier between, no store or atomic in the loop to any binding that may alias the source |
| vectorise four loads into a vec4 load | contiguous, 16-byte aligned, the whole vec4 in range (one guard covering all four lanes or a scalar tail), no aliasing store between |
| mask an index into its array (`& (len-1)`) | length a power of two AND the index already proved in range, so the mask is the identity in value; its gain is the compiler dropping its bounds check (measured 1.7x on 2026-10-11) |
| pad to the tile (with the dispatch plan) | padded buffer allocated, padding values neutral for the reduce |
| exploit symmetry (compute the upper triangle, mirror) | NOT a kernel rewrite: a thread writing its mirror cell breaks own-cell ownership; it is a dispatch-plan change (an upper-triangle grid plus a mirror pass) with its own ownership definition |

### 5.3 Banding: map over chunks, then reduce

A loop that accumulates over an axis with an associative combine
(`acc += f(k)`) becomes bands of size B: each band computes a partial (map),
a combine merges them (reduce). Precondition: associativity of the combine;
for floats the reorder changes rounding, which must stay inside the accuracy
tolerance, or the reduce uses pairwise or Kahan summation. Knobs, coupled:
the band size, the banded axis, the reduce strategy:

1. read-add-write into a resident buffer, one pass per band (the shipped Gram
   driver; needs the dispatch plan);
2. two passes: partials, then a reduce kernel (dispatch plan);
3. a tree reduction in workgroup memory inside one kernel (a block, section 2.2);
4. integer atomics, integer values only (WGSL has no float atomics).

### 5.4 Structural rules from the language

Each block has a few legal edits: wrap a statement sequence in a tile phase,
split a loop's body, add a guarded tail to a loop, add an `else`, replace a
loop body by an equivalent block from the magic symbol table.

### 5.5 Crossover

Swap blocks of the same shape and type between two parents (the typed subtree
crossover of phylu `wgsl_cross.rs`, which grafts subtrees of the same dual),
restricted to blocks whose free holes (the symbols they read) are bound in the
receiving parent, and carrying their group definitions with them.

## 6. Rules mined from model edits

### 6.1 Data

About 585 model replies were saved on 2026-10-11 with their prompts, which
hold the parent kernels. Many children were faster than their parents.

### 6.2 Method

1. Read parent and child with fuller's reader into trees.
2. Tree diff (a GumTree-style matcher): the smallest pair of changed subtrees,
   everything around them identical.
3. Anti-unification across pairs showing the same kind of change: the most
   specific common pattern, the differing leaves becoming typed holes. This is
   the magic table's generalisation step (literals and variables replaced by
   their types).
4. Test: apply the candidate rule to many other kernels; keep it when the
   children are legal and some are faster; record its precondition from where
   it failed.
5. Store: the rule goes into the replacement dictionary (fire counts,
   survival counts, pruned by evidence), persisted across runs.

### 6.3 Clean examples

Today's replies are often whole rewrites, so their diffs are large. A
targeted prompt asks for one change per reply ("make one performance change;
change nothing else"), which gives small diffs and clean rules. Models are
then a source of rules, not of every offspring.

### 6.4 Rule weights

Each rule keeps its record: fired, legal, faster, survived. The operator's
choice of rule (section 3, step 2) weights by that record, so rules that help
are fired more. Rules that never help are pruned.

## 7. Autotuning

The autotune practice (tracel.ai, "Autotune for GPU kernels") keys a cache
by the problem's shape (dimensions rounded to powers of two; whether the shape
is a whole number of tiles) and keeps the fastest variant per key. Here:

- the fitness measures each candidate on a **set of shapes** (a shape bucket);
- the dispatch plan (section 2.5) can carry **variants per bucket**, chosen at
  dispatch by the key;
- the evolved population becomes a tuner: its best individuals per bucket are
  the variant table.

## 8. Where it lives

| part | crate | today's anchor |
|---|---|---|
| tables, blocks, group column, interface, dispatch plan | fuller | `kingdoms/safewgsl`, `gen_tables.py`, `loader.rs` |
| reader (facts, groups), renderer | fuller | `safe_reader.rs`, `render.rs` |
| e-graph statement terms, rules, matcher | fuller | `src/wgsl/mutate.rs`, `src/ruleset/` |
| magic symbol table, replacement dictionary | fuller (data), phylu (use) | `homeotic.rs`, `population_dag.rs` |
| the mutation operator in the variation path | phylu | `wgsl_fit` variation (`make_new`, `wgsl_cross.rs`) |
| rule mining | fuller | new, reading the saved replies |

phylu depends on fuller; fuller never depends on phylu.

## 9. Measurement, with expectations stated before any run

**Pre-registered (revised after review finding 5).** Fixed before any run:
the corpus (`tests/corpus/safewgsl` reading kernels, populations
`logs/HGP/001/popB_g080` and `popA2_g006j`), the device (Apple M3 Max, Metal,
the wgpu and naga versions of the build), seeds 1 to 5, the correctness
oracle (the 128 cases plus the accuracy pairs), held-out shapes (ragged sizes
and the largest shape), the time limit and the evaluation budget. Rates are
reported with 95 % intervals and a failure taxonomy (which checker layer or
fitness stage). Ablations at equal attempted and completed evaluations:
uniform random legal knob settings; the same rules with repair or rejection
instead of the checker; typed crossover with and without parameters;
selection only (no variation); a small exhaustive grid over the knobs; and a
hand-tuned kernel. Baselines are from 2026-10-11.

| measure | baseline | expected after |
|---|---|---|
| knob mutation: legal on first evaluation | head mutation 16 of 92 | at least 95 % |
| meaning-preserving rewrite: correct on the device | n/a | at least 95 %; every failure is a rule bug (a missing precondition), fixed |
| block crossover: correct children on the device | 10 of 40 | 20 of 40 or more |
| improvement of the best per 1,000 evaluations | none in 1,040 | at least one step in the first 1,000 |
| best speed-up after 100 generations, no models | about 10.5x | 15x or more |

## 10. Tasks (each brief opens "Read docs/Rules.md, then:")

0. **First, the cheapest test of the riskiest assumption (review finding 6):**
   fix validation to the adapter's features and limits; then one first-class
   `block_x` parameter and one closed rule changing it and every dependent
   position. Run it over the saved populations and boundary shapes: render,
   validate, compile and run on the device, check ownership and barriers,
   measure legal share, diversity and speed. Only if it is close to 100 %
   legal and yields viable faster variants do the later tasks proceed.
1. Coverage: the full WGSL compute language in the kingdom (in progress).
2. Blocks: loop, if, tile phase, guarded store, tree reduction, band as block
   symbols with typed slots.
3. The group column and knob definitions in the tail; the reader assigns groups.
4. The interface and the dispatch plan slots.
5. The facts and device tables; the join evaluator.
6. Statement terms in the e-graph; the rule format; the matcher.
7. The catalogue of sections 5.1 to 5.4, each rule with a test that its
   children are legal on the corpus.
8. The operator in phylu's variation path; rule weights.
9. Rule mining from the saved replies.
10. Autotune: shape buckets in the fitness, variants in the dispatch plan.

## 11. Review

Codex adversarial review of this document (`codex exec`), its findings
answered in a revision, then Andrew's review. Nothing is built before.

## Open questions

1. Should a knob value outside today's sets be allowed (any integer the
   device permits), with the sets as the mutation's preferred values only?
2. Should meaning-preserving rewrites bypass the correctness cases (they are
   correct by construction) to save time, with a sampled check?
3. How much of the dispatch plan should the gene own before the first
   UltraDim target (the trellis beam), which needs a batched, persistent kernel?

## 12. Review findings and changes (Codex, 2026-10-12)

| # | finding | change |
|---|---|---|
| 1 | Joins, patterns and lookups do not prove legality; validation runs with every naga capability, not the adapter's | Section 1: a layered checker after every rewrite; mined rules untrusted until certified; the validation bug is task 0 |
| 2 | Shared tail definitions (positional hrefs) are unsound carriers for groups across crossover | Section 2.3: groups are first-class, closed parameter nodes; lowering to shared rows only via stable row ids later |
| 3 | Rewrite preconditions too weak (barriers, aliasing, atomics, vector bounds); symmetry breaks ownership; masking is only an identity | Section 5.2: preconditions over an effect, alias and barrier model; symmetry moved to the dispatch plan; mask's precondition and gain stated |
| 4 | An e-graph cannot hold ordered effects; large structural rule sets break the non-confluence rule | Section 4.2: statement rules on a typed control-flow matcher; egglog only on pure expressions, small capped families |
| 5 | Expectations not pre-registered; baselines and ablations missing | Section 9: corpus, device, seeds, oracle, held-out shapes, budget fixed; intervals; six ablations |
| 6 | The riskiest assumption is the group carrier | Section 10, task 0: one `block_x` parameter and one rule, measured, before anything else |
