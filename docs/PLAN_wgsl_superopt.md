# PLAN: the superoptimiser — a measured 10 % on one real kernel

## Context

**Objective.** A reproducible 10 % reduction in the GPU execution time
of phylu's `decode.wgsl` (`decode_main`), found by the genetic algorithm
searching the legal computation graphs the lineage representation can
express, with the measured pass time as the fitness and bit-exact
decode output as the gate; then the same on realistic workloads and
seeds.

**Approach.** The lineage work (`PLAN_wgsl_lineage.md`, built
2026-10-09) decides what is legal: which repeats may share one
definition and where, and which expressions may move. This plan adds the
moves worth searching over (share or recompute, hoist out of a loop,
cache a re-read load, rewrite an integer or bit expression), encodes a
kernel's inventory of legal moves as a genotype, measures each variant's
decode pass on the device with timestamp queries, and compares the
generated GPU code of the winner with the shipped kernel's so the gain
is explained, not just observed.

**Unique feature.** The compiler must assume any storage buffer may alias
any other, so it cannot hoist a load across a loop that stores to a
different buffer, cannot keep a re-read cell in a register across a
store elsewhere, and cannot share across a call. The reader's versions
know exactly which locations a loop or a call bumps, per invocation, so
the search can legally make moves Metal's optimiser is forbidden to
make. That is the hypothesis: the gain, if there is one, lives in moves
across the scaffold, not in algebra inside a statement (the six kernels
have ten float regions in all, and the folded rebuilds measured within
noise of shipped).

**Status.** Design, 2026-10-09, after the reviewer's direction
("performance improvement is now the primary objective … let the
genetic algorithm explore alternative legal computation graphs, with
measured GPU performance as its fitness signal"). Nothing built. The
measurement harness (`phylu/examples/kernel_time.rs`) times whole
generations; per-pass time is the first thing to build.

**Why this order.** Without a per-pass number the target is undefined:
decode is one pass of five in a 32.5 ms generation, and a 10 % gain on
it is invisible in the generation total. Without the move inventory the
genotype has nothing to vary. Without the generated-code diff a measured
gain could be the compiler's, not ours.

## 1. The measurement

- **Per-pass GPU time.** Timestamp queries (`wgpu::Features::TIMESTAMP_QUERY`,
  `ComputePassDescriptor::timestamp_writes`) around each pass in phylu's
  `Resident::run` (decode, eval, score, type, hff), resolved per
  generation, reported as the median over the run and the spread over
  three runs. phylu's change (they offered it); fuller's `Device::time`
  gains the same for the oracle's kernels. The target is 10 % of the
  decode pass's median, with the spread sized first: a gain inside the
  spread is not a gain.
- **The workload.** SRBench shape (population 600 + 200, 3 genes, head
  34), `feynman_I_29_16`, 300 generations, every stop off; three laws and
  three seeds for the validation of §5.
- **The gate.** Decode is a discrete kernel: its output (nodes, gene
  depths, gene ok, lengths) must be bit-identical to the shipped
  kernel's on the same population, every generation. The override sweep's
  golden checksum and sample gate stay as the final gate; a per-variant
  gate that runs in a second compares one generation's decode output
  buffers against the shipped kernel's (phylu records them; fuller
  compares words).

## 2. The legal moves

Each move is a choice the search may take; each is legal by the
decision in `legality.rs` or by a rule stated here and tested.

1. **Share or recompute.** Every repeat the decision accepts is a bit:
   fold it (one definition at its placement) or leave it inlined. The
   fold today takes every accepted repeat; the genotype decides per
   repeat. Sharing can cost registers, recomputation can cost ALU; the
   device decides.
2. **Hoist a single-use expression out of a loop.** `decide` already
   finds the earliest legal point for a set of sites; for ONE site inside
   a loop whose lineage has no header phi at that loop, the earliest
   point is before the loop. A hoist is a definition with one use, placed
   there. Candidates: every subtree of at least `min_ops` operators
   inside a loop body whose placement leaves the loop.
3. **Cache a re-read load.** A load repeated under one version is a
   1-operator repeat the fold ignores (`min_ops` 2). Admit loads as
   shareable definitions (`min_ops` 1 for `load.*` texts): a cell read
   twice between two stores to its buffer is read once into a `let`.
   Across a loop (the load's version unchanged by the loop) this is the
   hoist of §2.2 applied to a load: the move the compiler cannot make
   because it cannot prove the other buffers' stores do not alias.
4. **Integer and bit rewrites.** A new egglog ruleset over the `index`,
   `int`, `bits` and `logic` rows (the engine's actual workload; decode
   has no float region): `x * 2^k → x << k`, `x / 2^k → x >> k` and
   `x % 2^k → x & (2^k − 1)` for `u32`, `(x << a) << b → x << (a + b)` with
   the amounts' sum below 32, mask and shift identities, `x & x`, `x | 0`,
   `x ^ 0`, `x * 1`, `x + 0`, `select(a, a, c) → a`, `!(!c) → c`, and
   wrapping arithmetic reassociation (sound for `u32`, which wraps).
   Bit-exact by construction over the integers; still gated.
5. **Not in this plan.** Loop restructuring (fusion, unrolling, exchange)
   and workgroup-size changes: they change the scaffold, which the
   representation keeps fixed. Named so they are not confused with the
   moves above.

## 3. The genotype and the search

- **One DAG, not one text per variant.** The population of variants is
  held as one hash-consed DAG of the kernel's roots (the §7 requirement
  applied here): a genotype is a delta against it, the fold and the
  decision are maintained, and only the rebuild to text and the compile
  are per variant, because the compiler needs text.
- **Inventory.** For a kernel, the list of legal moves (§2.1–2.4) with a
  stable order: repeats by their text, hoists by their site, cached
  loads by their text, rewrites by region and variant. The inventory is
  computed once per kernel by fuller (`wgsl::superopt::inventory`).
- **Genotype.** One choice per inventory entry (a bit for 1–3, a variant
  index for 4). The shipped kernel is the all-zero genotype; the current
  folded rebuild is "every share on".
- **Phenotype.** The genotype applied to the roots (rewrites), then the
  fold restricted to the chosen shares, hoists and cached loads with
  their placements, then `rebuild_folded`: one WGSL text per genotype.
  A genotype whose text fails naga validation is refused and the
  inventory entry that caused it is reported (that is a legality bug).
- **Fitness.** The decode pass's median GPU time over N generations on
  the fixed workload, after the gate; a variant that fails the gate has
  no fitness and is not bred. Lower is better; ties broken by text
  length.
- **Search.** A small population (16) of genotypes, mutation flips one
  choice, crossover is uniform, 20 generations to start; each evaluation
  is one `kernel_time`-style run of a few generations with the variant
  through `PHYLU_WGSL_DIR`. Deterministic from a seed; the ledger records
  every genotype, its text hash, its gate result and its time.
- **What is run where.** fuller computes inventories and phenotypes and
  owns the search loop (`examples/wgsl_superopt.rs`); the timing is
  phylu's harness invoked per variant; one GPU process at a time, as
  always.

## 4. The generated code

For the shipped kernel and the best genotype: naga's MSL
(`naga::back::msl::write_string`, the same options wgpu-hal uses on this
platform), compiled with `xcrun -sdk macosx metal -S` to AIR text (the
Metal compiler's LLVM-level output after its own optimisation), and
diffed. The report states what changed beyond the source: a load
hoisted that Metal left in the loop, a shared value Metal recomputed, or
nothing (in which case the measured difference is noise and the gain is
withdrawn). The AIR is not the final GPU ISA; it is the furthest the
toolchain lets us look, and it is where the compiler's own LICM and CSE
have already happened.

## 5. Validation

The winning genotype on three laws (`feynman_I_29_16` and two others
phylu names) and three engine seeds, 300 generations each, the decode
pass time against shipped with the spread; the override sweep's golden
checksum on each; the generated-code diff for one. A reproducible 10 %
means: on every law and seed, the median decode pass is at least 10 %
below shipped and the two spreads do not overlap.

## 6. Order

The order follows the reviewer's priorities as given: loop-invariant
code motion, sharing versus recomputation, integer and bitwise
rewrites, memory access; a reproducible 10 % on decode measured as GPU
execution time; the generated code compared; then realistic workloads
and seeds; the genetic algorithm exploring legal graphs with measured
GPU time as its fitness. Each step is measured before the next is built.

1. **Per-pass GPU time.** Timestamp queries around each pass of phylu's
   resident chain; decode's own time; the spread sized on three shipped
   runs. The same instrumented fit records subtree reuse per generation
   for §7.
2. **Loop-invariant code motion** (§2.2) and **memory access** (§2.3,
   the cached re-read load, the load hoisted across a loop the compiler
   cannot prove safe): the inventory and the phenotype, unit tests per
   move, the oracle's 1,000 programs with random genotypes through the
   device, bit-exact.
3. **Sharing versus recomputation** (§2.1): fold and unfold as the two
   moves; register pressure against ALU, decided by the device.
4. **Integer and bitwise rewrites** (§2.4): the ruleset over the rows
   decode is made of, gated by the oracle.
5. **The search** (§3): genotypes over the inventory, measured decode
   pass time as fitness, bit-identical output as the gate, one GPU run
   at a time, a ledger of every variant.
6. **The generated code** (§4): naga's MSL and the Metal compiler's AIR
   for shipped and winner, diffed; the gain explained or withdrawn.
7. **Validation** (§5): three laws, three seeds, 300 generations,
   spreads not overlapping, the golden checksum held: the 10 %.
8. **The incremental population** (§7, §8), built on what the search
   needed anyway (the shared DAG of variants, the arena, the delta
   compiler): reuse between generations measured, Fold as an operator
   in the engine's loop, the compiled-against-interpreted experiment,
   compaction and the tail study. The measurements already taken (63 %
   repeated work in one generation; 80 ms to compile a band of a few
   hundred definitions) say it is worth doing; they do not reorder the
   reviewer's milestone.

## 7. Incremental folding: a general requirement (Andrew, 2026-10-09)

**The requirement, kingdom-independent.** Whatever the genes are, SR
expressions, WGSL kernels, SQL statements or Spark plans (the Dagnetic
Spark kingdom), a population under selection evolves slowly: most of
generation N+1 is generation N. So the evaluation of a population must
be a shared, incrementally maintained DAG with per-node memoised
results, not a per-individual recomputation, in every kingdom. The SQL
case makes it plain: a thousand statements over the same tables share
scans, filters and joins (Neumann and Moerkotte's share equivalence),
and the population's next generation re-uses almost all of them; an
optimiser that re-plans and re-costs every statement every generation
does the same work a thousand times over. The mechanism below is the
geneframe's, not the WGSL kingdom's; this plan states it for the first
two kingdoms it will serve.

A population evolves slowly: generation 101 shares most of its genes
with generation 100 (elites and survivors are identical, a child differs
from its parent by one gene or one subtree), so the fold, the decode and
the evaluation done at 100 are most of the work of 101, and today every
generation does them from scratch.

The concept: one shared DAG for the whole population's genes, hash-
consed so that equal subtree text is one node whichever individual
holds it, adjusted for change rather than rebuilt. A generation is then
a delta: new children add only their new nodes; a node keeps the values
it computed per row (the evaluation pass) so a child re-evaluates only
the nodes above its change; the fold is the DAG itself, maintained,
never recomputed. It is the lineage DAG without versions, since SR genes
read no memory; for kernels, where the population of variants shares
the scaffold and most definitions, it is the symbolic half of the
search (read, fold, decide, rebuild) made incremental, while the fitness
half (compile and timed dispatches) stays per variant.

**The design (Andrew): the cache is embedded in the population.** The
geneframe already holds the population as a table of chromosomes. Make
it a table of TWO row kinds: ordinary chromosomes, whose genes reference
shared definitions by href, and HOMEOTIC CHROMOSOMES, each of which IS
one definition, a gene tree with a root type, memoised row values and
a count of its referrers. The population is the union, so no separate
cache value exists; the geneframe is the state:

```
fold(population)        -> population'      (adds the definitions the new rows need)
eval(data, population)  -> population'      (memoises every row once, homeotic rows first)
```

**Fold is an operator.** In this form Fold is a variation operator
beside mutation and crossover, `population -> population'`,
deterministic and value-preserving on every ordinary row. Per
generation it ADDS a homeotic chromosome for each subtree the new
population repeats and no row yet defines (with its root type),
REWRITES ordinary chromosomes to reference those rows by href where the
typed projection admits it, and RE-SORTS the homeotic tail (below). Its
inverse, unfold (inline
a definition into its users), is an operator too, which is the
share-or-recompute choice of §2.1 for free. Fold sits in the schedule
after variation and before evaluation, so evaluation always sees current
definitions; Fold itself evaluates nothing.

**First version: let it grow (Andrew).** The tail starts empty and grows
by what each generation adds; no eviction, no frame, no maximum. Every
row keeps its referrer count per generation, and the run records rows
added per generation, the growth curve, the use histogram and, for any
definition that goes from unused back to used, how long it was unused.
The maximum, the frame and the ordering below are set from that study,
not before it.

**Later: the homeotic tail and its open reading frame (Andrew).** Nothing is
retired by rule. The homeotic rows are one tail of fixed length, sorted
every generation by current use (referrer count, ties by age then by
hash, so the order is a function of the population and deterministic).
A definition nobody uses sinks one place per generation as used ones
rise, and falls off the end when it reaches it. The top of the tail is
the open reading frame: the rows that are expressed, whose values are
memoised in the device arena; its length is the memory budget in rows.
Below the frame a row keeps only its text and type, so a subtree that
returns finds its definition and only its values are a miss. This gives
bounded memory with no eviction policy, graceful forgetting instead of a
nursery timer, and the replacement dictionary for free: persisting the
tail across runs is saving the table.

Consequences: evaluation is uniform, every row once per data version,
homeotic rows in dependency order first, which is phylu's tail-per-level
dispatch applied across the population instead of inside an individual;
use is a definition's fitness, measured, not judged (a useful definition
outlives the individuals that made it); the
href space is the table, an href names a homeotic row, the device
decodes homeotic rows into a shared node arena and ordinary genes point
into it with the GeneRef leaf that exists today; a homeotic row carries
its root dual, so a reference is type-checked by the typed projection
like a terminal. The earlier form of this section (a cache value passed
beside the population) is kept below as the functional contract it
still satisfies: the population IS that value.

There is no batch mode and no incremental mode: one function body. A
population with no homeotic rows (or no memos) makes the call a batch
(every node a miss); last generation's population makes it incremental
(reused definitions hit, only the nodes above a child's change are
computed); a data version bump makes every memo stale and the next call
a batch again. A run is a fold over generations with the population as
the accumulator.

The homeotic rows hold the DAG (node hashes, child pointers) and, per
row, a slot of its row values with the data version they were computed
on; the slots are a device-resident arena allocated once, one per row
of the open reading frame, released when a row sinks below the frame,
so a definition that returns is just a miss. The arena is owned by the
population value: whoever holds the population holds the memory,
dropping it frees everything, and no session-global state exists.

Determinism is the contract and a test: a homeotic row changes only
whether a value is recomputed, never what it is. The test runs one
generation with the homeotic rows stripped and with them present and
asserts bit-identical fitness and ordinary population; the same seed and
the same starting population give the same run; and the golden checksum
holds with homeotic rows on. A difference is a fault in the memo keys,
never tolerated as noise.

What it costs: cached values are rows times live nodes (an SRBench-
shaped fit, about 50,000 live nodes at 1,000 rows, is about 200 MB) and
must live on the device to pay off. What decides it, measured first from one
fit before anything is built: the fraction of subtrees per generation
already present in the previous generation, and the eval pass's share
of the generation (from the per-pass timestamps of §1). If reuse is
high and eval is the large pass, the incremental DAG is the next plan
for phylu's engine, written as a general mechanism (a population DAG
with hash-consed nodes, a per-generation delta, per-node memoised
results keyed by node and data version) that the SR, WGSL and SQL
kingdoms instantiate; it is recorded here so the kernel search, the
engine's own speed and the Dagnetic Spark kingdom are designed on one
idea.

## 8. The experiment that tests the idea (Andrew: "build a kernel for the population itself")

The 10 % of §1 is the reviewer's milestone for decode, not a test of
the incremental idea. The idea is that a population's work can be
COMPILED instead of interpreted, and stay compiled across generations
because little changes. The study that proves or bounds it:

**Hypothesis.** A kernel generated for the population's shared DAG
(the homeotic tail), compiled by Metal, evaluates a generation faster
than phylu's interpreter (decode and eval over node arrays); and because
generation N+1 reuses most of N, the compile is amortised, so the total
per generation is lower. The number that decides it is the break-even
churn: how many new definitions per generation the compiled path can
absorb before recompiling costs more than interpreting saves.

**Data and problem.** Three SRBench Feynman laws at the CASCADE2 shape
(population 600 + 200, three genes, head 34), 300 generations, three
seeds, the official data windows. Same seeds, same laws, two
evaluators, bit-identical populations required.

**The two arms.** Interpreted: the current chain, per-pass timed.
Compiled, incremental: the homeotic tail emitted as one WGSL kernel that
writes definition values into the device arena (the kingdom's rebuild
already writes WGSL from trees), recompiled only when the tail changes;
the ordinary chromosomes stay node arrays whose leaves reference the
arena through GeneRef.

**Measured per generation.** Rows added to the tail; whether a
recompile happened and its cost; the eval time of each arm; the reuse
fraction. Derived: the speedup per evaluation of compiled over
interpreted, compile time as a function of tail size, the break-even
churn.

**What proves it.** Compiled-incremental total per generation below
interpreted on all three laws and seeds, spreads sized and not
overlapping, populations bit-identical. If compile time grows faster
than the tail amortises, the result is the measured break-even and the
idea is bounded, not proven; that is a result.

**The unknown that decides it early, measured first.** Metal's compile
time for a kernel of tens of thousands of expressions: emit a kernel
for one real population's definitions, compile it, time it. If that is
seconds, the compiled arm must be split into pieces compiled separately
(one kernel per band of the tail, recompiled only when its band
changes), and the plan changes before anything else is built.

**Measured 2026-10-09, on Andrew's instruction ("run a hard problem,
unsolved for 90 s").** `strogatz_predprey1`, the sweep's config, seed
7014, 90 s: 3,535 generations, 14.1 M individuals, not solved (1−R²
7.6e-3, test R² 0.992), population of 4,000 individuals × 3 genes
exported (`phylu/logs/HARD_predprey1/population.json`; `feynman_I_26_2`,
also run, solved at generation 79 in 13.4 s, so it is no longer hard).
Hash-consed, the final population has 288,918 subtree occurrences,
37,345 distinct subtrees, 7,012 of them repeated with at least one
operator (182,037 of the occurrences, 63 %), 6,836 with two or more;
5,694 distinct genes of 12,000. A kernel of those 6,836 definitions
(`kingdoms/wgsl/samples/predprey1.definitions.wgsl`, 466 KB, straight
line, one `let` each, values into an arena) compiles on the device in
8.9 s cold and 0.45 s when Metal's cache has it; naga's own parse,
validate and write take 0.5 s of that. A generation takes 25 ms. So the
compiled arm cannot be one kernel recompiled when the tail changes: a
cold compile costs 350 generations. The arm must be banded (kernels
over slices of the tail, each recompiled only when its slice changes)
or restricted to the stable, heavily reused definitions, with the
compile time per band measured against its size before the design is
fixed. That is the plan change this section said would follow; it
follows.

**Compile time against kernel size (measured 2026-10-09, same
population, cold compiles on Apple M3 Max):** 50 definitions 79 ms,
200 → 86 ms, 1,000 → 490 ms, 3,000 → 2.7 s, 6,836 → 8.9 s. Flat at
about 80 ms up to a few hundred definitions (the fixed cost of a
pipeline), then worse than linear. So the append-only form (Andrew:
"compile things previously uncompiled, and run both executables
together") works at a band of a few hundred definitions: one delta
kernel per band, its dependencies read from the arena, the live kernels
dispatched in order, compaction in the background. The fixed 80 ms is
three generations, so a delta is compiled every few generations or on
another thread while the interpreter carries the new rows, never on the
generation's own path.

This section governs the order: the compile-time measurement came
first (above), and the decision point of §6 step 1 reads three
numbers: decode's pass time, the reuse fraction per generation, and
the compile time per band.

## What could make this fail, stated now

- The compiler may already make every move the inventory offers on
  decode, in which case the diff of §4 is empty and the honest result is
  "no move in this inventory beats Metal on decode", with the inventory
  enlarged (§2.5) in a further plan.
- Register pressure: sharing on a GPU is not free; the search decides,
  which is why share-or-recompute is a choice and not a default.
- Noise: per-pass timing on a shared GPU with the WindowServer; the
  spread is sized before any claim, and runs are one at a time.
