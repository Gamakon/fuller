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

1. Per-pass timestamps in phylu's resident chain (phylu), the spread
   sized on shipped (three runs); fuller's `Device::time` gains
   timestamps too. The SAME instrumented fit also counts subtree reuse
   per generation (§7). **Decision point, from those two numbers:** if
   eval is the large pass and reuse is high, the incremental population
   DAG (§7) is worth more than any 10 % on decode and is built first, as
   its own plan for the engine; the kernel search then targets the
   kernels that remain after that change, not the ones it replaces. If
   decode is the large pass or reuse is low, this plan continues as
   written.
2. The inventory and the phenotype in fuller (§2.1–2.3, §3): shares as
   choices, hoists, cached loads; unit tests per move on the oracle's
   kernels; the oracle's 1,000 programs with random genotypes through the
   device (bit-exact, as the folded rebuild was).
3. The integer ruleset (§2.4), gated by the oracle.
4. The search loop with the gate and the timing per variant; decode
   first.
5. The generated-code diff of the best genotype.
6. The validation of §5 and the README tables.

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

Consequences: evaluation is uniform, every row once per data version,
homeotic rows in dependency order first, which is phylu's tail-per-level
dispatch applied across the population instead of inside an individual;
liveness replaces eviction, a homeotic chromosome with no referrers is
dead and one with many is valuable, a fitness measured by use that
joins the replacement dictionary and nursery ideas (a useful definition
outlives the individuals that made it and persists across runs); the
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
on; the slots are a device-resident arena allocated once, addressed by
row, released when a row dies (no referrers, past its nursery), so a
definition that returns is just a miss. The arena is owned by the
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

## What could make this fail, stated now

- The compiler may already make every move the inventory offers on
  decode, in which case the diff of §4 is empty and the honest result is
  "no move in this inventory beats Metal on decode", with the inventory
  enlarged (§2.5) in a further plan.
- Register pressure: sharing on a GPU is not free; the search decides,
  which is why share-or-recompute is a choice and not a default.
- Noise: per-pass timing on a shared GPU with the WindowServer; the
  spread is sized before any claim, and runs are one at a time.
