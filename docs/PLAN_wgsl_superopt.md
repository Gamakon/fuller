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
  decode pass's median, measured as paired, randomised A/B runs with
  confidence intervals (§8): a gain inside the interval is not a gain.
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
  held as one hash-consed DAG of the kernel's roots (the population DAG
  requirement, now `docs/PLAN_population_dag.md` §1, applied here): a genotype is a delta against it, the fold and the
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
pass time against shipped as paired, randomised measurements with
confidence intervals (§8: same inputs, the two variants in random order per
repeat, ten repeats); the override sweep's golden checksum on each; the
generated-code diff for one. A reproducible 10 % means: on every law and
seed, the paired difference in decode pass time is at least 10 % of
shipped, with its confidence interval wholly beyond 10 %.

## 6. Order, as tasks

Every task below follows `~/dev/minkymorgan/qdrant/docs/Rules.md`: no
stubs, no simulated results, no failovers, warnings fixed at the root,
every change committed with a factual message, real data, tests that
fail loudly and run to completion, wall-clock time reported for every
run, timeouts set from the run's own requirement (a fit's budget plus
its build, never a round number), temporary scripts kept in tmp/. The
order follows the reviewer's priorities. The wider project, for every
task: the WGSL kingdom turns a compute kernel into a chromosome whose
sharing and rewrites are legal by construction (memory versions,
`legality::decide`, the folded rebuild, the oracle, all built and
measured in `kingdoms/wgsl/README.md`); this plan uses that to make
phylu's decode kernel measurably faster by evolutionary search, with
the device's own time as the fitness and bit-identical output as the
gate. Tools throughout: fuller (`src/wgsl/*`, `examples/wgsl_oracle.rs`,
`examples/wgsl_read_kernel.rs`), phylu (`examples/kernel_time.rs`, the
resident chain in `src/evolve/resident.rs`, `scripts/wgsl_override_sweep.sh`),
wgpu 0.20 on Metal, naga 0.20, `xcrun metal`. One GPU process at a time,
launched only by the fuller session, announced, under nohup with a log.

1. **Read docs/Rules.md, then: per-pass GPU time.** Purpose: the target
   is decode's own time, which no number today gives (a generation is
   32.5 ms over five passes). Architecture: `wgpu::Features::TIMESTAMP_QUERY`
   and `timestamp_writes` on each compute pass of `Resident::run`, resolved
   per generation into the KERNEL_TIME line of `kernel_time.rs`;
   fuller's `Device::time` gains the same for the oracle's kernels.
   Measure: decode's median pass time and the spread over three shipped
   runs, plus subtree reuse per generation (for `docs/PLAN_population_dag.md`,
   where it was measured as that plan's task 1, §2a). Done when the three
   runs and their spread are in the README with wall-clock times.
2. **Read docs/Rules.md, then: loop-invariant motion and memory access
   as legal moves.** Purpose: the moves the compiler cannot make, since
   it must assume storage buffers alias and our versions know they do
   not. Architecture: `wgsl::superopt::inventory` in fuller lists, per
   kernel, every single-use subtree inside a loop whose placement by
   `legality::decide` leaves the loop (§2.2) and every load repeated
   under one version, `min_ops` 1 for `load.*` texts (§2.3); the
   phenotype applies a choice vector and calls `rebuild_folded`. Tests:
   one unit test per move on the oracle's hand-written kernels; the
   1,000 generated programs with random choice vectors through the
   device, bit-exact. Done when both pass and decode's inventory is
   counted in the README.
3. **Read docs/Rules.md, then: sharing versus recomputation.** Purpose:
   sharing costs registers on a GPU and may lose; the device decides.
   Architecture: each accepted repeat becomes a choice (fold or inline)
   in the inventory; unfold is the inverse operator (§2.1). Tests: the
   all-zero vector reproduces the shipped text, the all-one vector the
   folded rebuild, bit-exact on the oracle. Done when decode's
   choices are counted.
4. **Read docs/Rules.md, then: integer and bitwise rewrites.** Purpose:
   decode is integer code; the Math rules reach nothing in it.
   Architecture: a new egglog ruleset in `src/ruleset/` over the `index`,
   `int`, `bits` and `logic` rows (§2.4), used by `mutate.rs` with the
   same opaque-leaf mapping; bit-exact by construction and still gated
   by the oracle's criterion. Tests: a unit test per rule; the 1,000
   generated programs' integer regions mutated and run on the device.
   Done when the pass rate and the count of decode's integer regions are
   in the README.
5. **Read docs/Rules.md, then: the search.** Purpose: let selection
   find what is fastest among the legal graphs. Architecture: §3, a
   genotype per inventory entry, population 16, mutation flips a
   choice, uniform crossover, 20 generations, fitness the decode pass
   time of task 1 through `PHYLU_WGSL_DIR`, gate the bit-identical
   decode output, a ledger (genotype, text hash, gate, time, wall
   clock) per variant, `examples/wgsl_superopt.rs`. Timeout per variant:
   its warm-up plus its measured generations, not a round number. Done
   when the ledger and the best genotype are in the README.
6. **Read docs/Rules.md, then: the generated code.** Purpose: a gain
   must be explained or withdrawn. Architecture: §4, naga's MSL for
   shipped and winner with wgpu-hal's Metal options, `xcrun -sdk macosx
   metal -S` to AIR, a diff read line by line, no sampling. Done when
   the diff and its reading are in the README.
7. **Read docs/Rules.md, then: validation.** Purpose: reproducible means
   across workloads and seeds. Architecture: §5, three laws, three
   seeds, 300 generations each, shipped against the winner as paired,
   randomised measurements with confidence intervals (§8), the override
   sweep's golden checksum on each. Done when the table shows the decode
   pass at least 10 % below shipped on every law and seed, the confidence
   interval of the paired difference wholly beyond 10 %, or states by how
   much it falls short.
8. **Read docs/Rules.md, then: the incremental population** is
   `docs/PLAN_population_dag.md`, which runs first; this plan's tasks
   1–7 follow its decisive experiment.

## 7. The incremental population DAG: its own programme

The design and the measurements that were §7 and §8 of this plan are
now `docs/PLAN_population_dag.md`, a research programme of its own that
the external reviewer's second review (2026-10-09) put BEFORE this one:
"the 10 % WGSL experiment proves a compiler capability; the incremental
population DAG could become a genuinely differentiated computational
engine". This plan proceeds after that programme's decisive experiment,
on the kernels that remain.

## 8. The reviewer's two concerns on this programme, answered

- **Compiler aliasing must be demonstrated, not assumed.** Before task 2
  builds the hoisting moves, a demonstration: a kernel with a load of
  buffer A inside a loop that stores to buffer B, compiled to AIR with
  `xcrun metal -S`, read line by line for whether the load was hoisted;
  the same with a `let` hoisted by hand. If Metal already hoists it, §2.3
  loses its first claim and the inventory is measured without it.
- **Integer rewrites carry WGSL's semantics explicitly.** Each rule in
  the §2.4 ruleset states its domain: `x / 2^k → x >> k` and `x % 2^k →
  x & (2^k − 1)` for `u32` only (signed division truncates toward zero
  and shifts do not); shift amounts masked to 31 so `(x << a) << b` is
  `x << (a + b)` only when `a + b < 32`; wrapping add and multiply
  reassociate for `u32`; the platform's `x / 0` deviation means a
  rewrite may not introduce a division whose divisor can be zero. Each
  rule has a device test on the edge values from the oracle's hand set.
- **Statistics.** Timing claims use paired, randomised A/B measurements
  with confidence intervals (same inputs, variants in random order per
  repeat, ten repeats), not non-overlapping spreads.

## What could make this fail, stated now

- The compiler may already make every move the inventory offers on
  decode, in which case the diff of §4 is empty and the honest result is
  "no move in this inventory beats Metal on decode", with the inventory
  enlarged (§2.5) in a further plan.
- Register pressure: sharing on a GPU is not free; the search decides,
  which is why share-or-recompute is a choice and not a default.
- Noise: per-pass timing on a shared GPU with the WindowServer; the
  spread is sized before any claim, and runs are one at a time.
