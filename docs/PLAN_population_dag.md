# PLAN: the incremental population DAG — one shared, incrementally compiled DAG for a whole population

## Context

**Objective.** Evaluate a population of chromosomes as ONE shared DAG,
maintained incrementally across generations, so that work repeated
within a generation and between generations is done once; prove it on
real populations against the current GPU interpreter, with bit-identical
results, and measure the saving.

**Approach.** The geneframe holds two row kinds, ordinary chromosomes
and homeotic chromosomes (population-wide shared definitions with a
root type, memoised row values and referrer counts); Fold is a variation
operator that adds definitions and rewrites references; evaluation is
one function with no modes; the compiled arm is banded (delta kernels
reading an arena). One decisive experiment isolates the gain of sharing
from the gain of compilation: A the current interpreter, B the shared
DAG evaluated without compilation, C the shared DAG with incrementally
compiled bands, on the same exported populations.

**Unique feature.** The population is the cache: the shared definitions
are rows of the same table as the individuals, selected by use, typed
by their root dual, persisting across runs. No kingdom-specific cache:
SR genes, WGSL kernels and SQL or Spark plans (the Dagnetic Spark
kingdom) instantiate one mechanism.

**Status.** Design 2026-10-09 (Andrew's direction through the session;
the external reviewer's second review reversed the priorities: this
programme before the WGSL superoptimiser, `PLAN_wgsl_superopt.md`).
Measured so far on a real unsolved population (`strogatz_predprey1`,
90 s, 3,535 generations): 63 % of subtree occurrences belong to
repeated subtrees within one generation; a kernel of a few hundred
definitions compiles in 80 ms, of 6,836 in 8.9 s; a generation is
25 ms. Not yet measured: reuse between generations, dispatch overhead
of many small kernels, memory, the achievable saving. Nothing built.

**Why first.** Decode is one pass of five; a 10 % on it is small in the
generation total. Sharing across the whole population reaches decode
and evaluation alike. The reviewer: "a fundamentally larger
optimisation opportunity, assuming the reuse and memory-access
economics hold" — which is what the experiment below decides.

## 1. The design (moved from PLAN_wgsl_superopt.md §7)

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

**Later: the homeotic tail and its expressed frame (Andrew).** Nothing is
retired by rule. The homeotic rows are one tail of fixed length, sorted
every generation by current use (referrer count, ties by age then by
hash, so the order is a function of the population and deterministic).
A definition nobody uses sinks one place per generation as used ones
rise, and falls off the end when it reaches it. The top of the tail is
the expressed frame: the rows that are expressed, whose values are
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
of the expressed frame, released when a row sinks below the frame,
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


## 2. The first measurements (moved from PLAN_wgsl_superopt.md §8)

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
interpreted on all three laws and seeds, as paired randomised
measurements with confidence intervals (§4; the arms and laws are
§4's), populations bit-identical. If compile time grows faster
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

The compile-time measurement came first (above); the order is §5 of
this plan.


## 2a. Task 1 measured: reuse between generations (2026-10-09, 11:02–11:04)

`strogatz_predprey1`, the sweep's config, seed 7014, 90 s: 4,110
generations (21.9 ms each), the population exported at generations
0 and 1 modulo 250 (phylu 6bd8cb63, `EVOLVE_EXPORT_POP_DIR`/`_EVERY`;
33 files, 601 MB, `phylu/logs/REUSE_predprey1/pop/`). Sixteen
consecutive pairs, counted on the host (wall clock 3 min):

| what, per pair (g, g+1) | mean over 16 pairs |
|---|---|
| distinct subtrees of g+1 present in g | 24 % (6,500 of 27,000) |
| REPEATED subtrees of g+1 (two or more occurrences, one or more operators) present in g | 72 % (3,650 of 5,050) |
| new repeated subtrees per generation | 1,431 |
| genes of g+1 textually identical to a gene of g | 65 % (7,800 of 12,000) |

The spread across the sixteen pairs is narrow (22–26 %, 66–76 %,
1,130–1,820, 62–70 %); the numbers are stable over the whole fit.

**Reading.** Two thirds of the genes survive a generation unchanged,
and nearly three quarters of the shared definitions do; but a quarter
of the working set of definitions is new every generation, about 1,400
of them. At the measured compile costs (§2: 80 ms for a band of a few
hundred, 490 ms for 1,000) compiling each generation's new definitions
would cost roughly 0.5 s against a 22 ms generation, even banded, even
off the generation's path: the compiled arm as "compile what is new"
cannot keep up with this churn. What survives: (a) variant B, the shared
DAG with memoised values and no compilation, which is unaffected by
compile cost and inherits 72 % of its definitions' values each
generation; (b) a compiled arm restricted to definitions that have
proved STABLE, those present for k generations (the use-sorted frame's
top), compiled once and reused for as long as they live, with the
unstable quarter interpreted. The decisive experiment of §4 keeps its
three variants with C redefined as (b), and the tail study (task 7)
gains the number that sets k: the survival curve of a definition by
age.

## 2b. Task 2 measured: dispatch overhead (2026-10-09, 11:06)

`wgsl_oracle --dispatch`: one submission holding N compute passes of
one workgroup each, four pipelines cycled so every pass pays a pipeline
switch as banded kernels would, median of 15 submissions, Apple M3 Max:

| passes per submission | median ms | µs per pass |
|---|---|---|
| 1 | 1.27 | 1,267 |
| 10 | 1.28 | 128 |
| 100 | 3.69 | 37 |
| 1,000 | 22.6 | 22.6 |

A submission costs about 1.3 ms whatever it holds, and each further
pass about 22 µs. Against a 22 ms generation: ten bands are free, a
hundred cost 2.4 ms (11 %), a thousand cost a whole generation. So the
compiled arm may hold up to a few tens of bands between compactions,
and the interpreter's own per-level dispatches (one per tail level per
generation today) sit on the same curve.

## 2c. Task 3 measured: the shared DAG with memos, on the host (2026-10-09, 11:11–11:13)

`src/population_dag.rs` (fold into one hash-consed DAG; homeotic rows
= repeated nodes doing work; memo keyed by node, data version, type and
input binding; level schedule; nothing evicted) and
`examples/population_dag_reuse.rs`, over the sixteen exported pairs of
`strogatz_predprey1` with 200 real rows of the law's data, host only:

| per generation (mean of 16) | operator evaluations |
|---|---|
| an interpreter (every operator of every gene, occurrences and all) | 99,661 |
| the shared DAG, fresh (one per distinct subtree) | 33,167 (33.3 %) |
| the shared DAG with last generation's memos | 24,433 (24.5 %) |

The memos avoid 26 % of the DAG's own work (72 % of the definitions
survive, their values hit); the sharing within a generation does the
larger part, three times fewer operator evaluations before any memo.
Memoised values are bit-identical to a fresh evaluation on every pair
(the determinism test, held on real data). The DAG holds about 50,000
nodes for 12,000 genes, 5,000 of them homeotic, 1,400 of those new each
generation. Host time (a per-row Rust loop, not the device): 176 ms
fresh, 161 ms with memos; the host cost is dominated by hashing and
copying, so the device measurement (variant B, task 4) is the one that
counts. What task 3 establishes: the work a population-wide DAG saves
on this population is a factor of four in operator evaluations, a
quarter of it from memos across generations, with no change in values.

## 2d. Two things the survey of the documents found missing (2026-10-09)

**An href names a stable row id, never a tail position.** The homeotic
tail is re-sorted by use every generation; if a reference pointed at a
position, every re-sort would rewrite every referrer. So a homeotic row
has an id assigned when it is interned and kept for its life
(`population_dag::PopulationDag` already does this: the node index is
the id, the use rank and the level are separate columns), the device's
GeneRef leaf carries that id, and the sort permutes a rank column
only.

**Names.** "Tail" and "open reading frame" already name Karva's gene
tail and coding region in `Fuller_System.tex` and `SPEC_fuller_gpu.md`.
In this plan the population-wide structure is always the HOMEOTIC TAIL,
in full, and its expressed, memoised top is the EXPRESSED FRAME; the
Karva terms keep their meanings.

## 3. The reviewer's three risks, answered as design

- **Dispatch overhead (A).** Hundreds of small kernels may cost more than
  interpreting. Measured before the band size is chosen: the cost of N
  empty-ish dispatches in one submission on this device (fuller's
  `Device::time` with N bands), against the interpreter's per-level
  dispatches today. Band size and compaction interval are set from
  that curve, not assumed.
- **Memoisation validity (B).** A memo is keyed by the node's structural
  hash, the data version, the numerical type of its root (the dual),
  and the input binding (which columns and which window of rows the
  leaves read); two nodes equal in text but differing in any of these
  are two memos. Execution context cannot differ for SR genes (no
  memory); for kernels it is the version and placement lineage already
  carries. A test constructs a pair that differs only in each key
  component and asserts separate memos.
- **Order of execution (C).** The homeotic tail's sort by use is a
  liveness order, not an execution order. Execution follows a separate
  dependency schedule: definitions by level (a definition's level is
  one more than its deepest referenced definition), levels dispatched
  in ascending order, exactly phylu's tail-per-level dispatch today.
  The two orders are kept apart in the data (use rank; level), and a
  test asserts no definition is evaluated before one it references.

## 4. The decisive experiment

On the same exported populations (predator-prey and two more laws,
three seeds, populations exported every generation for 300
generations):

| variant | what runs |
|---|---|
| A | the current GPU interpreter (decode → eval over node arrays) |
| B | the shared DAG evaluated without compilation: homeotic rows memoised in the arena, ordinary rows interpreted against it |
| C | the shared DAG with its STABLE definitions compiled (present k generations; §2a), bands reading the arena, the unstable rest interpreted |

Measured per generation, wall clock stamped: total ms; GPU memory
(arena and buffers); compilation cost (C only, when it happens); exact
numerical agreement (bit-identical populations across A, B and C, every
generation). Reported as paired, randomised A/B/C measurements with
confidence intervals (the same population, the three variants in
random order per repeat, ten repeats), never as non-overlapping
spreads. This isolates sharing (A→B) from compilation (B→C).

## 5. Order, as tasks

Every task follows `~/dev/minkymorgan/qdrant/docs/Rules.md` (no stubs,
real data, warnings fixed, every change committed, wall-clock reported,
timeouts from the run's requirement). The wider project for every task:
the geneframe engine (fuller the symbol/typing/share layer, phylu the
GPU evolution engine) evaluating populations as one shared DAG. Tools:
fuller `src/homeotic.rs`, `src/geneframe.rs`, `src/wgsl/*`; phylu
`src/evolve/{resident,device,gene_export,checkpoint,fold_to_dag}.rs`,
`examples/{evolve_fit,kernel_time}.rs`; wgpu 0.20 on Metal. One GPU
process at a time, launched by the fuller session, announced, logged.

1. **DONE 2026-10-09 (§2a). Read docs/Rules.md, then: reuse between generations.** Purpose:
   the number the programme rests on. Architecture: `evolve_fit` with
   the population exported every generation (`EVOLVE_EXPORT_POP` per
   generation, a phylu change), predator-prey, 90 s; a host count
   (fuller example over the exports) of the fraction of each
   generation's subtrees present in the previous generation, and of new
   definitions per generation. Done when the curve is in the README
   with wall-clock times.
2. **DONE 2026-10-09 (§2b). Read docs/Rules.md, then: dispatch overhead.** Purpose: risk A.
   Architecture: `Device::time` over N trivial kernels in one submission
   for N in 1, 10, 100, 1,000, against one kernel; the curve sets band
   size and compaction interval. Done when the curve is in the README.
3. **DONE 2026-10-09 (§2c, host side; the device arena is task 4). Read docs/Rules.md, then: the arena, the memo keys and the level
   schedule in fuller.** Purpose: variant B's machinery. Architecture:
   §1 and §3; homeotic rows with structural hash, root dual, data
   version, input binding; levels computed from references; the arena
   device-resident, owned by the population value. Tests: the key-
   component pair test, the level-order test, determinism (homeotic
   rows present or stripped, bit-identical). Done when the tests pass
   on the exported population.
4. **Read docs/Rules.md, then: variant B in phylu.** Purpose: sharing
   without compilation. Architecture: Fold as an operator in the
   generation loop (adds, rewrites, re-sorts), evaluation reading the
   arena for homeotic rows, ordinary rows interpreted; checkpoints
   carry the homeotic rows. Tests: golden checksum held; A and B
   bit-identical every generation. Done when B runs 300 generations on
   predator-prey with the gate held.
5. **Read docs/Rules.md, then: variant C, the banded compiler.**
   Purpose: compilation's own contribution. Architecture: §2's
   append-only bands, the kingdom's rebuild emitting a band's kernel
   reading the arena, compiled off the generation's path, bands
   dispatched by level, compaction in the background. Tests: C
   bit-identical to A and B. Done when C runs the same 300 generations.
6. **Read docs/Rules.md, then: the decisive experiment of §4.** Done
   when the paired A/B/C table with confidence intervals, memory and
   compile costs is in the README, with the result stated plainly
   whichever way it falls.
7. **Read docs/Rules.md, then: the tail study.** Growth, use counts,
   return depths over the runs of task 6; the frame and the maximum set
   from them.
