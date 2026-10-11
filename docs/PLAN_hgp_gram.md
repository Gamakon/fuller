# PLAN: hallucino-genetic programming (HGP) on the Gram kernel — experiment 001

## Context

**Why.** Three runs of the typed-tree search today learned nothing: no fitness
improved over 31 generations, because a pipeline's output fitness has no slope
between all-right and one-step-wrong (an index off by one gives an entirely
wrong table; the 1,024 probes confirmed it: the shipped kernel passes all, the
shipped kernel with its multiply broken fails all). Andrew's proposal: let a
language model hallucinate whole kernels, and let the compiler, the probes and
the HFF judge; vary the survivors by fuller's e-graph rewriting (a strong
editor given a starting program), by model edits and by crossover.

**Objective.** On Blue World's weighted Gram `H = ΦᵀΦ` (the harness, criterion,
87 reference pairs and 1,024 probes already exist), measure: of 32 Sonnet
kernels how many parse, validate, compile, pass the probes and the pair
criterion, and are faster than shipped; per mutation operator, the share of
offspring that stay correct; the champion's paired speed-up on held-out shapes
after 500 evaluations. The research question: does model diversity plus
selection beat the model alone (generation 0) and the rule-only editor alone.

**Approach.** Staged fitness: a candidate climbs stages (parses → validates and
compiles → correct on probes and pairs → fast), selection by highest stage then
the HFF angle over the performance columns within the top stage; invalid
candidates kept with diagnostics for one model repair. The model proposes and
never judges. The committed design is `fuller/docs/PLAN_hgp_gram.md` (84a7c2b).

## What exists and is reused (fuller, feature `gpu,wgsl`)

- `src/wgsl/gram.rs`: `Gram::compile` / `compile_rendered`, `build(phi, d, m, limit)` with
  GPU timestamps, `check` (criterion), `phi_of(Kind, …)`, `reference`, `effective_buffer_limit`.
- `src/wgsl/probes.rs`: `probes(seed)` (1,024, families A–E), `run_probes`, `distance`, `PASS`.
- `src/wgsl/reader.rs` + `chromosome.rs` + `mutate.rs` + `scaffold.rs`: `read(text)`,
  `chromosome_with_roots`, `mutants_of_roots(roots, min_ops, k, iters, with_loads)` (the
  e-graph family), `rebuild_folded` → text. The reader covers If, Switch, Loop, Barrier and
  the workgroup address space (reader.rs:360–607, 687), so model-written tiled kernels read.
- `examples/wgsl_gram_evolve.rs`: the cost bound (`render::cost`), the watchdog, the
  ledger pattern, the pair splits (`splits`), the final split scoring and paired timing.
- `claude` CLI at `/Users/andrewmorgan/.local/bin/claude`: `-p`, `--model sonnet`,
  `--output-format json`, `--system-prompt`, `--no-session-persistence`; no temperature flag.

## The experiment in eight steps (Andrew, 2026-10-10, in plain words)

1. **Understand the problem.** The kernel multiplies a matrix of items by its own transpose, in bands. The contract (bindings, band rules), the reference answers (87 large pairs, 1,024 small test cases) and the kernel to beat exist. Done.
2. **A population.** Individuals are kernels.
3. **The initial population is drafted by an AI**, one kernel per individual: 32 calls to Sonnet, each given the contract and a strategy. Running (generation 0).
4. **Parse every draft into Karva genes.** This is why the drafts are not kept as text: as typed genes, crossover and random mutation stay well-formed, and fuller's share-equivalence scan over the population finds the subtrees many drafts have in common and moves them into homeotic genes, the reusable parts evolution composes from. That is where the search becomes creative instead of choosing among drafts. **Built 2026-10-10** (fuller 64cca68, 11614a5; phylu c49cce06): the safeWGSL kingdom (`kingdoms/safewgsl`, 126 duals, 561 rows) spells only kernels that cannot crash or hang (bounded loop counts, the loop condition produced by one row, index provenance for every store, masked array indices, statement levels with no barrier under a condition); `src/wgsl/safe_reader.rs` reads a draft into it, `src/wgsl/safe_karva.rs` encodes the tree as one head gene and homeotic tail genes and builds the kernel back from decoded genes. Measured below.
5. **Evolve from that population** in phylu's engine.
6. **Fitness of an individual:** gene → kernel text (the renderer, exists) → run it (the fitness evaluation: parse, compile, test cases, reference pairs, timing; exists) → the HFF over the measurements.
7. **Mutate some individuals with fuller** (the rewriting rules; exist for the arithmetic, the structure needs step 4).
8. **Mutate some at random, cross over, hold tournaments:** phylu's operators.

Order of work: finish generation 0 and read the drafts; extend the genome and write the reader (step 4); hand phylu the population with the fitness call; run steps 5–8.

## Design

### The fitness evaluation (a constraint, then the objective)

| stage | test | measured as |
|---|---|---|
| 1 | `naga::front::wgsl::parse_str` | pass, or the first error line |
| 2 | naga validation; Metal pipeline (`Gram::compile_with` through an explicit layout) | pass, or the first error |
| 3 | the constraint's 128 test cases (a fixed, seeded sample across families A–E); then the full 1,024 and the fixed pairs for the top 8 of the population | probes passed; pair criterion max ratio |
| 4 | paired timing against shipped on 3 held-out shapes, 3 rounds, random order | speed-up per shape, geometric mean, worst shape, spread |

Within stage 4 the HFF angle over: speed-up on each shape (reciprocal, shipped = 1),
worst-shape speed-up, spread over rounds; all against the shipped kernel. Within stage 3,
probes passed. A stage-4 champion is never displaced by a lower-stage candidate.

### The model calls (`hgp::llm` in the example)

- Command: `claude -p --model sonnet --output-format json --no-session-persistence --system-prompt <contract>` with the task on stdin; the JSON's `result` field is the reply; fences stripped; the kernel must start with a fixed header comment `// grid x y bx by` that the harness parses into `render::Grid` (the dispatch rule `ceil(d/(x·bx)) × ceil(r_rows/(y·by))`).
- Contract (system prompt): the three bindings and `Cfg` verbatim, item-major Φ, band semantics (`r_rows`, `r_off`, read-add-write into `h`), the dispatch rule, the shipped kernel as reference, "return only the kernel".
- Diversity: a strategy per call from a fixed list of 10 (baseline, register block 2×2, 4×4, workgroup tile of items, vector loads of four columns, symmetric half, split accumulators, unrolled item loop, two combined, unlike the shipped kernel) plus a seed phrase; each prompt and reply saved to `logs/HGP/<run>/calls/<id>.{prompt,reply}.txt` with parentage.
- Mutation B (model edit): the kernel, its stage and measurements, one instruction from {reduce memory traffic, change the accumulation, tile differently, vectorise, drop the symmetric half's work, fix this error: <diagnostic>}.
- Mutation C (crossover): two parents' texts and measurements, "combine the named strategy of each"; typed subtree crossover in tree form when both parents read.
- Repair: one per invalid candidate, the exact diagnostic quoted.

### Mutation A: the e-graph editor (Andrew: "a strong editor given a starting program")

`read(text)` → per function `chromosome_with_roots` → `mutants_of_roots(roots, min_ops=2, k=8, iters=4, with_loads=true)` → each mutant's roots through `rebuild_folded` → text → stage 1. Also the `--variants` share subsets (`homeotic::unfold_slot`) as structural variants. Probability 0.35 at the start; a model edit 0.35, crossover 0.15, fresh 0.10, repair 0.05; the rates measured per operator and reported (adaptive weights later, not in 001).

### The population as a library: share-equivalence and homeotic genes (Andrew)

A large population of model-written kernels is scanned for share-equivalence the way
`src/population_dag.rs` folds a generation of SR genes: every kernel read through the
reader into the kingdom's form, every subtree hash-consed across the population, the
frequently seen subcomponents (an item-stride address `t·d + c`, a tile load into
workgroup memory, a register-blocked accumulation, a mirrored store) promoted to homeotic
rows with a stable id and a referrer count. Two uses, both measured:

- **Report** per generation: distinct subtrees, the share rate across kernels, the top 20
  components by referrers with their text, and which of them the correct kernels carry
  that the incorrect ones lack (the components that matter).
- **Evolve from components**: the popular components become terminals of the typed
  sampler and the subtree-replacement mutation (a mutation draws a popular subtree of
  the demanded dual instead of a random one), so the random and typed search composes
  from parts the model has shown to be useful. This is the bridge over the cliff for the
  tree search that learned nothing today, and it is measured against it: the same search
  with and without the component terminals, correct kernels per 100 evaluations.

Task order: the scan and the report in task 2 (it needs the population); evolving from
components is task 2b after the report shows which components the correct kernels share.

### The loop (`examples/hgp_gram.rs`)

Population 32; generation 0 = 32 fresh calls; later generations keep up to 8 stage-4 elites
(fewer when fewer reach stage 4; the slots stay empty) and make 24 offspring by the operator probabilities; 500 evaluations = 32 + 19.5 × 24, so 20 generations;
ledger row per candidate (gen, id, operator, parents, stage, diagnostic, probes passed, pair
ratio, speed-ups, angle, path to text); the run restartable from the ledger; the cost bound
and watchdog of `wgsl_gram_evolve` reused; one GPU process; the GPU's speed re-measured
first (shipped build 2.35 s at bench, 15:43; 8.4 s at 19:14 — a machine restart is needed
before stage-4 timing means anything; stages 1–3 can run regardless).

## Reviews (2026-10-09 evening) and what changed

Two adversarial reviews, both read every line of the plan and the code it reuses:
Codex (`logs/HGP/codex_review.md`, 19 findings) and an independent Claude reviewer with no
context from the day (`logs/HGP/claude_review.md`, 7 sections). Accepted, with the change
each causes:

| finding (both unless marked) | change |
|---|---|
| The HFF angle in `wgsl_gram_evolve.rs` was the angle to the diagonal: 0 for ANY constant vector, blind to magnitude (Claude 4.1). This, not only the cliff, is a cause of "learned nothing". | Replaced by phylu's definition (`hff.wgsl` 212–213): cos θ = 1 − mean(x²). Committed. |
| Mutation A (the e-graph rules) rewrites only f32 arithmetic regions; it cannot reach tiling, blocking, loops, stores, the grid (Claude 1.2, Codex 6). It is not the "strong editor" for this problem as the code stands. | A becomes a measured pilot (experiment step 2), not 0.35 of the budget; its expected effect on speed is about zero. |
| "Fails all probes" was asserted in the plan while the test asserted otherwise (Codex 1, Claude 1.10). | Measured and recorded in the test: a global break fails 1,024/1,024; the loop one item short fails 306/1,024, every one a probe touching the last item. The probes discriminate local breaks. |
| The 128-probe constraint is unstratified and can omit every band-forcing probe (Codex 2); probes are too small for tiled kernels (Claude 4.6); the probe pass is a mean over cells and lets one cell be 1 % wrong (Claude 4.7); stage 3's ranking mixes denominators (Codex 3). | Admission is stratified across families with every band-forcing D probe included; probes added at d ≥ 130, m ≥ 2 × the largest tile; pass is the per-cell maximum; a kernel is "correct" only after all 1,024 probes and all training pairs; stage 4 requires it. |
| No cost bound exists for model text; the watchdog cannot pre-empt the GPU; compile is unguarded; device errors can panic (Claude 1.1, 6.1–6.4). | Every candidate evaluated in a child process; a naga-IR pass refuses loops not bounded by `cfg.m`, `cfg.d`, `cfg.r_rows` or a literal ≤ 4,096 and nested bounds over m × 64; the first run is the smallest probe under a 2 s deadline; error scopes for Validation and Internal and an uncaptured-error handler. |
| The model call runs with tools and CLAUDE.md discovery, so the proposer can read the judge (Claude 1.9). | `claude -p --bare --model sonnet` with no tools, from an empty directory; the auth path verified first. |
| The grid header is unchecked against `@workgroup_size` (Codex 8, Claude 1.6). | x and y come from naga; only bx, by from the header; a mismatch is refused at stage 2. |
| Symmetric-half strategies cannot work across output-row bands under this driver (Claude 1.5). | Removed from the strategies and the edit instructions; the band constraint stated in the contract. |
| Timing: three rounds against a 12 % spread selects noise; the timed builds' H is unchecked; the held-out shapes are selected on; no shipped-vs-shipped null; no GPU acceptance test (Codex 10–11, Claude 2.3, 2.4, 4.3–4.5). | Every timed build's H checked against its reference and bit-identical over two builds; the shipped-vs-shipped null every batch; the GPU accepted only when the shipped bench build is within 10 % of 2.35 s; selection shapes and report shapes disjoint; the champion re-timed in a fresh session with 5+ rounds. The clock is `gpu_ms` (timestamps, non-zero at the shapes used, resolved after completion). |
| The research question has no compute-matched control (Codex 12, Claude 2.1). | The comparison arm is best-of-N fresh calls at the same model-call and GPU budget. |
| The share-equivalence scan needs cross-kernel alpha-renaming, let inlining, index AC-canonicalisation, statement-level components and per-kernel referrer sets, none of which the reader or `population_dag` provide (Codex 16–17, Claude 5.1–5.6). | The scan is deferred until the reader's statement increment and a canonicaliser exist; task 2 keeps only the text-level report (distinct normalised texts, strategies); task 2b waits. |
| Budget arithmetic and wall clock wrong (Codex 14, Claude 1.11, 3.3). | Corrected below. |
| `Kind::Cancelling` did not cancel (Claude 2.8). | Redescribed and retested: the pair doubles every off-diagonal term; sign handling is what it tests. |

Not accepted: Codex's reading that the probes have "no slope" is answered by the measurement above; the `#[allow(dead_code)]` on `timing` (Claude 1.7) is removed with the next edit of the example.

## The smallest experiment (replaces the loop as the first thing built)

0. **Positive control and timing null**, after the machine restart: one tiled kernel written by hand (16 × 16 workgroup, 4 × 4 register block, item tile in workgroup memory) and the shipped kernel through 1,024 probes, all training pairs and the extrapolation pairs; then paired timing, 5 rounds, at d = 2,048, m = 65,536, beside shipped-vs-shipped rounds. If the tiled kernel is not at least 1.3× faster under this driver, the speed question has no headroom here and the experiment stops.
1. **Generation 0 alone**: 32 isolated `--bare` calls, texts saved; stages 1–2; every validating kernel through all 1,024 probes and all training pairs; every correct one timed as in step 0. Report per strategy: parsed, validated, correct, faster; distinct texts among the 32. If none is correct and faster, one repair pass over the 32 is the only next step.
2. **Operator A on real inputs**: the shipped kernel and the best generation-0 kernel through read → `mutants_of_roots(2, 8, 4, true)` → rebuild; distinct texts counted; three mutants timed against their parent. Expected: no distinct speed.
3. **Only if 0 and 1 pass**: best-of-N fresh calls against the evolutionary loop at equal budgets (N = 100), the champion re-timed on unseen shapes and fresh rounds against the null; the hypothesis holds only if the loop's champion beats best-of-N by more than the null's spread.

## Step 0 measured (2026-10-10, `logs/HGP/ladder_blocked.log`, ledger `logs/HGP/ladder.tsv`)

The machine restarted overnight; the orphaned kernel is gone and the shipped build is 1,015 ms at the bench size (yesterday's 2,348 ms was under load). Two kernels written by hand:

| kernel | correctness | speed-up over shipped, median of 5 paired rounds (null shipped/shipped) |
|---|---|---|
| tiled: 16 × 16 workgroup, 4 × 4 register block, 32-item tiles in workgroup memory (vec4 groups) | 1,024/1,024 probes; 87/87 pairs | d 2,048: 8.08 (null 1.01); d 4,096: 7.99 (0.99); d 8,192: 7.76 (1.00); geometric mean 7.9. Earlier readings of 0.33× and 0.59× were taken while another GPU consumer had the chip at 96 % (shipped itself at 1.3–1.4 s); a kernel with barriers loses far more under contention than the shipped kernel, and the shipped-against-shipped null did not show it. Hence the GPU acceptance test now in the fitness evaluation: stage 4 runs only when shipped alone builds within 12 % of 1,015 ms. |
| blocked: 16 × 16 workgroup, 4 × 4 register block, two 16-byte loads an item straight from phi, no workgroup memory | 1,024/1,024 probes; 87/87 pairs (train, validation, extrapolation); worst error equal to shipped's | d 2,048 m 65,536: 3.77 (null 0.99); d 4,096 m 16,384: 3.98 (1.00); d 8,192 m 4,096: 4.25 (1.02); geometric mean 4.0, worst shape 3.77 |

The headroom exists (the rule was 1.3×), so the experiment continues. On a quiet chip the workgroup-memory tiled kernel is the faster of the two (7.9× against 4.0×); both structures are worth asking the model for. The fitness evaluation (`examples/wgsl_gram_ladder.rs`) ran the whole of stages 1–4 in 115 s; 1,024 probes take 5.7 s on the healthy GPU (100–166 s on yesterday's).

## Step 4 measured (2026-10-10, `wgsl_safe_convert logs/HGP/001/gen0 --out logs/HGP/001/safe --karva logs/HGP/001/safe/karva`)

| measurement | expected | measured |
|---|---|---|
| drafts of 32 read into the safe grammar, rendered and validated by naga | 20–28 | 27 (k05 helper functions; k06 loop test `t + 4 > m`; k08 array constructor; k16, k26 loop bound `m & ~3`) |
| reader trees row-consistent before the conforming pass (every child the dual its parent's row demands) | all | 8 of 27; 19 needed a count-to-index, scalar-to-vector or index-shift conversion inserted, or a four-lane compose nested |
| one layout for the population: head length, tail length, tail slots, genes per individual | head ≤ 64 | head 32, tail 97, 48 slots (k04 needs 43), 49 genes |
| genes per individual with a dual code on every live token | all | all (phylu's typed loader excludes none) |
| individuals whose genes decode on the host with no fallback and rebuild to the converted kernel's text, byte for byte | 27 | 27 (fuller test `the_converted_population_round_trips_through_karva`, phylu test `the_safe_population_decodes_and_rebuilds_to_its_text`) |
| converted kernels compiled and run on the device (feasible under the the constraint's 128 test cases) | 25–27 | not run: the GPU is held at 100 % by the kernel orphaned on 2026-10-10 (`ioreg` Device Utilization 100 % with no process of ours); needs the machine restart |

The dumps are `logs/HGP/001/safe/karva/kNN.json`; the converted kernels `logs/HGP/001/safe/kNN.safe.wgsl`. The run that follows the restart: `wgsl_fit --kingdom safe --dumps fuller/logs/HGP/001/safe/karva` (phylu), generation 0 first, every row evaluated once; expected: the 27 feasible, HFF as generation 0 measured them.

## Engine runs measured (2026-10-11, `logs/HGP/001/fit_engine3..5`, seed 3, population 30 = 26 drafts + 4 drawn)

Fitness: the HFF angle over the speed-up columns (lower is better); an infeasible kernel scores π/2.

| run | variation | generations | best HFF (gen) | feasible of 30 per generation | new feasible offspring per generation |
|---|---|---|---|---|---|
| fit_engine3 (control) | engine operators | 4 (stopped) | 0.3283 (3) | 5–7 | 1–3 of 26 |
| fit_engine4 | engine operators + champion mating + the gate | 13 (hung, see below) | 0.3230 (1), flat to 13 | 18–27 | 6–23 |
| fit_engine5 | head-only: deduplicating head mutation + champion mating + the gate | 35 (4 h budget) | 0.3204 (25); 0.3228 (15); 0.3254 (1) | 17–27 | 7–23 |

Champion mating (Andrew: an infeasible individual is a failed kernel, not diversity) turned one to three
feasible offspring a generation into about twenty. The gate (an infeasible offspring replaced, in turn, by a
recross, the champion crossed with it, the champion with a head mutation, until feasible) still leaked 3–12
rows a generation. Cost: an infeasible evaluation is 2.0 s (128 cases, one GPU round trip each), a feasible
one 8 s; a generation 5–8 min.

fit_engine4 froze at 02:45Z on the CPU: champion mating assembled a tail whose genes read later slots twice
at every slot, 2^48 nodes to unfold; `homeotic::unfold` is now bounded at 65,536 operators (fuller 495d288).

**The re-timing (07:22Z, d 1024, m 16384, 7 rounds each, `logs/HGP/001/retime_*.log`): the fitness
differences among the top kernels are timing noise.**

| kernel | HFF in the run | GPU ms median (min–max) | speed-up over shipped |
|---|---|---|---|
| shipped | — | 64.56 (64.40–73.83) | 1 |
| best draft, generation 0 | 0.3300 | 10.66 (10.53–23.52) | 6.06× |
| champion, generation 15 | 0.3228 | 10.61 (10.57–10.84) | 6.08× |
| champion, generation 25 | 0.3204 | 10.57 (10.49–22.27) | 6.11× |
| offspring at the floor | 0.3287 | 10.49 (10.46–10.56) | 6.15× |

All four are the same speed within 1 %, every round correct.

Repeated at the fitness's own three shapes (07:26Z, d 2048 m 65536, d 4096 m 16384, d 8192 m 4096, seven rounds,
`logs/HGP/001/retime_shapes_*.log`, 0 failing rounds):

| kernel | ms median at d 2048 / 4096 / 8192 | speed-up over shipped, re-timed | speed-up the run's single round gave |
|---|---|---|---|
| shipped | 1009 / 1005 / 1013 | 1 / 1 / 1 | — |
| best draft, generation 0 (HFF 0.3300) | 150.0 / 146.3 / 152.9 | 6.73 / 6.87 / 6.62 | — |
| champion, generation 15 (0.3228) | 150.0 / 155.4 / 152.1 | 6.73 / 6.47 / 6.66 | 8.55 / 6.82 / 7.54 |
| champion, generation 25 (0.3204) | 155.1 / 146.3 / 152.3 | 6.51 / 6.87 / 6.65 | 7.81 / 8.00 / 7.84 |
| offspring at the floor (0.3287) | 149.4 / 145.4 / 152.8 | 6.75 / 6.92 / 6.63 | — |

The four kernels are within 4 % of one another at every shape; the run's single paired round per shape had
put the same kernels anywhere from 6.0× to 8.7×, a spread four times the real differences. The ranking the
fitness produced among the top kernels is timing noise. Thirty-five generations found many kernels as
fast as the best model-written draft and none measurably faster: the "floor" at 0.3287 that the offspring
kept hitting, and the "new bests" below it, are the timing spread of equal kernels. Before any further run
the fitness needs more rounds per shape (or the median of repeated timings) so that a real 5 % gain is
distinguishable from noise, and the 128-case screen needs its one-submission form (cases resident on the
device, one command buffer) so a generation costs seconds, not minutes.

## Step 5: model-repaired variation (Andrew, 2026-10-11 08:00Z) — the plan

**Objective.** Make crossover produce feasible, new kernels that combine two parents' ideas, and
make every infeasible offspring a repair job rather than a discard, so that over time the symbol
table grows with what works, the share of feasible genes rises, and ideas mix and get tested.

**Approach.** The standard engine (islands, tournament selection, elites, pump, cohorts) stays. Its
own gene-slot crossover stays off (it mixed kernels of different geometry into garbage: 6 % feasible).
Variation becomes three model-mediated operators plus the head mutation, all through the gate:

1. **Shared-structure crossover.** For an offspring whose two tournament parents were both feasible
   and distinct, fuller finds the largest sub-trees the parents share (`safe_karva::shared_sections`,
   the population fold over a forest of two). The brief carries the contract, both parents' kernels,
   the six fitness columns defined and each parent's measured values, each parent's problem in one
   line, the shared structure in words and s-expressions, and the size bound of the layout. The
   model writes one kernel around the shared structure. Proven once: k15 × k07 → feasible, HFF
   0.3245 against the parents' 0.3303 and 0.5196, accuracy of the better parent kept exactly.
   **Two children a pair** (Andrew, 08:40Z: "crossover is often about the mixing of genes producing
   surprising outcomes"): two model calls in parallel, one focused on the union (the parents
   integrated around the shared section, into the row), one removing the union (parent A's
   processing outside the shared middle kept, the middle reinvented, into the mate's row). Both
   are checked after read-back: did the union keep the largest section, did the reinvention
   remove it; both counted in the generation line.
2. **Repair.** An offspring that scores infeasible goes to a model with its text, its failure detail
   (the first failing case or the compile error), its parents and the contract: "make it correct,
   keep its technique". The repaired text re-enters the gate. Only after the repair budget is spent
   does the gate fall back to the champion with a head mutation.
3. **Fresh kernels** at the pump, as now, with the same brief quality (columns defined, the
   population's best values stated as the bar).

**The model mix** for every call: Sonnet 70 %, Opus 20 %, Fable 5 %, Codex 5 % (`claude -p --model`,
`codex exec --skip-git-repo-check`), drawn per call from the run's seed; the ledger records which
model wrote each kernel and whether it was feasible, so the mix can be judged per model. No timeout
on a model call, ever; a kill guard only around GPU work. Every prompt and reply saved under
`out/calls/`.

**What has to change underneath, in this order.**

| step | change | test |
|---|---|---|
| a | the layout: `wgsl_safe_convert --slots N`; the drafts re-encoded at the slot count a model kernel needs (the k15 × k07 child needs 64 at head 32; take 96 for headroom); `n_genes` checked through the fold's reserved band, `mate_of`, the linker mask | the population round-trip test at the new layout; the child encodes |
| b | the table (Opus's review, `logs/HGP/001/opus_table/reply.md`): the slot-0 reference off-by-one; unexpressed gene positions written as the empty token, never as untyped names; the kingdom's whole vocabulary minted before any draft (every row as a typed function instance, every input-less row as a typed variable, the closed literal sets `flag`, `wgdim`, `blk`, `unroll`, `lit64`, `lit4096` in full, a fixed grid for the open literal types) | after minting, the 26 drafts add no symbol; the k15 × k07 child's remaining needs counted and named |
| c | **learning across runs**: every feasible model-written kernel's dump is written to a library directory (`logs/HGP/001/library/`), loaded at the next launch beside the drafts, so the table and the seed population grow with what worked | a run's library loads and round-trips |
| d | fuller `llm`: the model mix, the two new briefs, the saved calls; Codex as a caller | a replay path: a saved reply in place of the call drives brief → read-back → encode → table → dedup deterministically on `opus_cross/reply.wgsl` |
| e | the hook: operator 1 for tournament pairs (budget `--model-cross N` a generation), operator 2 in the gate (budget `--repairs N`), declines by reason, the generation line gains attempted / feasible / declined per operator and per model | the replay test; a two-generation run before any long one |
| f | **the fitness must be able to judge it**: `FitnessContext` takes timing rounds per shape and uses the median (`--timing-rounds`, default 1 so nothing changes until set). Re-timing showed one round scatters equal kernels by 15 %; the experiment "does this crossover work" is unanswerable at that resolution | the re-timed kernels rank equal at 5+ rounds |

**Expectations before the run.** Feasible model crossovers: 40–70 % of calls (one of one so far).
Repairs that make an infeasible offspring feasible: 30–50 %. Model cost: 15–60 s a call, so a
generation with 4 crossovers and 4 repairs is 3–8 min. Declines for table reasons: under 10 % after
step b (was 75 %). Whether the champion improves beyond the drafts: unknown; that is the experiment,
and only readable with step f on.

**Codex's adversarial review (`logs/HGP/001/codex_review/reply.md`, 08:02Z) and what it changed.**
Budgets defined: crossovers and repairs each capped a generation, repairs also per offspring (2); a
row whose repairs fail goes on through the gate as before (recross, champion crossing, champion head
mutation), counted as a gate failure, which Andrew set as the gate on 2026-10-11 01:10Z. The shared
section is checked after read-back (`occurrences_of`) and reported, not enforced. The mix is drawn
from one generator seeded by the run, in call order; a declined call counts against the budget and
is named by model and reason. The library loads only dumps of the population's layout and names the
rest. Timing rounds stay a flag, but the launch configuration below sets 5. Success criterion for
"the crossover works": at least 40 % of model crossovers feasible over 10 generations, and at least
one child under both its parents' HFF at 5 timing rounds. The one-submission 128-case screen and the
engine-side table growth are not in this step. No timeout on model calls stands (Andrew's rule).

**Built (2026-10-11 08:30Z; fuller 096f193, phylu ddaaf147), not run.** Steps a–f above are in the
code: `wgsl_safe_convert --slots`, the drafts at `logs/HGP/001/safe96`; the table's vocabulary
(`TypedPopulation::mint_kingdom`), the slot-0 reference, unexpressed positions as the empty token;
the library; `llm` with the mix, the Codex route, saved calls and the two briefs; `shared_sections`;
`FitnessContext::timing_rounds`; the operators in `wgsl_fit`. Tests: the 26 drafts add no symbol
after minting; Opus's child replays into the layout adding none; prose is declined; k15 and k07
share their geometry; the mix draws 70/20/5/5; the briefs carry the measurements; the median.
Not yet tested: a live generation with the operators (needs the GPU and model calls).

**The launch configuration proposed** (Andrew launches): `run_engine.sh fit_engine6 30 3 14400
--dumps logs/HGP/001/safe96/karva --variation head --pump-every 2 --timing-rounds 5 --model-cross 4
--repairs 4 --repairs-per-offspring 2 --mix true --llm 4 --library logs/HGP/001/library`.
Population 30 (intake 20, champion 10), elites 2, seed 3, 30 generations or 4 h. Expected cost: 4
crossovers + up to 4 repairs + (every other generation) 4 fresh kernels at 15–60 s each, plus the
gate's evaluations at 2–8 s and 5 timing rounds a feasible kernel: 10–20 min a generation. A
two-generation smoke run first (`--gens 2 --seconds 1800`), to see the operators' lines.

**Launch.** Nothing launches until Andrew says so; the full configuration is stated first.

## Step 6: the complete kingdom (Andrew, 2026-10-11 09:05Z: "Fix the kingdom!!!! I want a fully complete kingdom")

**Objective.** No correct, safe kernel the contract allows is ever declined by the grammar. The
smoke run of 08:52Z declined four model children in four, and five of the 32 drafts never read; each
was a report that the kingdom is incomplete, which I had filed as model error and routed to repair.
That broke Rules.md (no failovers, no graceful diversions, root causes not symptoms).

**Complete means two lists, kept apart.**

*Gaps, to be spelled (the job):*
- a loop bound that is any u32 expression (`cfg.m & ~3u`, `cfg.m / 4u`, `(cfg.d + 63u) / 64u`, a local), not only `cfg.m`, `cfg.d`, `cfg.r_rows` or a literal ≤ 64 (k16, k26, two pump declines);
- a loop test written as `t + 4u > m` (k06) or with the operands reversed, normalised to `counter < bound`;
- a loop step above 64 for the item loop;
- a vector component indexed by a variable, rendered through a helper so naga's constant-index rule holds (one Sonnet child);
- an array constructor initialising a local array (k08), lowered to private-array stores;
- helper functions whose body is lets and one `return expression` (k05), inlined at the call;
- more registers (32 f32, 16 vec4, 32 u32, 8 bool) and private arrays (a second vec4 array).

*Refused by design, kept (each protects a property the types were built for):*
- a barrier under a thread-dependent condition (a WGSL uniformity error, a hang on the device);
- a store to `h` at a cell this thread does not own (a data race between threads);
- a return under a condition that is not the leading guard (makes every later barrier non-uniform);
- more than two workgroup arrays of a kind (Apple's 32 KB threadgroup limit: the four arrays are 24 KB);
- `switch`, `continue`, pointers, f16, subgroups, atomics (outside the contract).

**Expression bounds and cost.** A loop bounded by an expression is rendered as written, never capped
(a cap would compute the wrong thing silently on a large shape). The cost estimator counts such a
bound as the worst shape's size (8,192 for one derived from `cfg.d` or `cfg.r_rows`, 65,536 from
`cfg.m`, 8,192 otherwise), so a kernel that would be slow is refused at the cost bound before
dispatch, by its true cost.

**Acceptance.** `tests/corpus/safewgsl/`: the 31 drafts and the 10 saved model replies with a
manifest (`reads` / `by-design` / `invalid`); `safe_reader::tests::the_corpus_reads_as_its_manifest_says`
prints the table and fails on one mismatch. Of the 10 replies, 3 are invalid WGSL (two with
unbalanced braces, one indexing a vector value by a variable, which naga refuses); the other 7 and
all 31 drafts must read. The 26 converted drafts must still round-trip byte-equal.

**Done (2026-10-11 10:05Z; fuller a5a012b, 0821264, cb5c537).** The corpus: 31 of 31 drafts and 7 of 7
valid model replies read and render valid WGSL; the 3 remaining replies are naga's own refusals
(two with unbalanced braces, one indexing a vector value by a variable). 127 duals, 629 rows (was
126 / 561). What was added, by the kernel that needed it:

| gap | kernels | spelled as |
|---|---|---|
| registers beyond 16 / 8 / 16 / 4 | model kernels (reg.f7..f10) | 32 f32, 16 vec4, 32 u32, 8 bool registers; a second vec4 private array |
| a loop bound that is any expression | k16, k26, pump declines | `u32.boundx` from `bound.of_index`; `loop.items_boundx` (+ strided, unrolled) at the top level, `loop.tilex_<rung>` below it; costed at the worst shape, never capped |
| a loop test not written `t < bound` | k06 (`if (t + 4u > m) { break; }`) | normalised to `t < bound − k` (`index.sub_floor`), with `<=`, reversed operands and the negated break forms |
| an item-loop stride above 64 | — | `lit4096` on the strided item loops |
| a vector component by a variable | k08, a Sonnet child | `shape.at`, rendered through a helper that copies into a `var` (naga's constant-index rule) |
| an array constructor into a local array | k08 | one private-array store per element |
| helper functions | k05 | inlined at the call when the body is lets and one return |
| a vector composed from vectors | k05 (`vec4(vec2, vec2)`) | the parts taken apart into lanes, then the step compose |
| a thread's rows held in a private array; `select(0u, x, ok)` | k26 | the array recorded as a row array when filled at a tile counter; any sum of wid/lid/offset terms |
| a high mask `m & ~3u` | k16, k26 | `index.round_down`; an integer literal above 2^24 is refused by name |
| a second item loop over the same counter (a remainder loop) | k05, k26 | `loop.items_cont_<bound>` (+ strided): `for (; t < bound; …)`, the counter continuing; found by the GPU check, which had the rebuilt forms at 28/128 cases while the kernels as written passed |
| a value the inference types two ways | a Fable child (local_invocation_index), two repairs (a loop counter) | the kingdom's preferred form chosen (index first), conform converts |
| the kingdom's own block-offset spelling `min(k, B − 1) * W` | two repairs of rendered kernels | read back as the block row it is, so every kernel the kingdom renders reads again |

**The GPU sweep (11:40Z, `logs/HGP/001/corpus_sweep.tsv`): 45 of 46 valid corpus kernels reach the same
verdict as written and rebuilt; the 46th, k08, is wrong as written (a race on shared vec4 elements) and
infeasible both ways.** The sweep found five more kingdom bugs the reader and the validator could not: a
vec4 register's lane written or read by index (k04, k14, k18, k24, k27, k28, two children) was spelled as
a private array and left the register untouched; a block factor of 8 (a Sonnet child) was snapped to 4;
the vec4 workgroup arrays were 256 long where a Fable child needed 512 (now 512: two f32 arrays of 2,048
and two vec4 arrays of 512 are exactly Apple's 32 KB and run); the remainder loop (k05, k26); the
kingdom's own offset spelling. Every one is a row or a reader case now, with the corpus as the test.

Checked on the GPU (`wgsl_eval_dump`, as written beside rebuilt on the same 128 cases and pairs):
k05 and k26 feasible both ways after the continuing loop; the corpus sweep (`logs/HGP/001/corpus_sweep.tsv`)
scores all 46 valid kernels both ways. The cost estimator counts an expression-bounded item loop by its
passes over the items at the worst shape (`cfg.m * cfg.d` is 8,192 passes, refused). The library's five
genes re-encoded at the current kingdom (the strided item loop's stride dual changed).

Byte-equal round trip of the 31 drafts at both layouts; every brief restates the grammar's limits as
they now are.

**Order** (smallest change first, a commit and the corpus test after each): registers and arrays;
expression bounds and loop-test normalisation; the dynamic vector component; the array constructor;
helper inlining. Then the tables regenerated, the drafts re-encoded (`safe96`), the hall of fame and
the library re-encoded, since a kernel's genes name rows and types by name and survive, but anything
that pinned a count does not.

## The cost gate measures, the estimate only guards (2026-10-11)

**Why.** A Metal dispatch cannot be cancelled, so a kernel that does far too much work can hold the GPU
for minutes (2026-10-08 it froze the machine). The static cost estimate kept such kernels out, but on
2026-10-11 it was wrong four times (estimator or reader over-counts) and kept out correct kernels 8-9x
faster than shipped. Andrew asked why we filter on an estimate; the answer is to measure.

**What.** Every build in the fitness is timed (wall time, submit to completion). The time is modelled as
a floor plus a rate per item-cell, `ms = floor + rate * d * d * m`: the floor is the median of the 128
correctness cases (at most 8e4 item-cells, all submission cost), the rate the least-squares slope of every
build so far through that floor, clamped at 0 (`gram::Meter`). The builds run smallest work first: the
correctness cases, two timing rungs (d 128 m 64 and d 256 m 512, timed and not judged), then the six
accuracy pairs across the three sets, then the timing shapes; no step is more than 32x the work of a
measured build. Before each build the time is predicted; a kernel predicted over the limit, or past the
evaluation's budget, is infeasible with stage `slow`, the detail naming the shape, the prediction and the
rate, and is not dispatched there. The repair brief explains `slow` to the model.

**The limits, from the watchdog.** `wgsl_fit`'s watchdog ends an evaluation at 30 s plus 20 s per timing
round (`gram::BASE_BUDGET_MS`, `ROUND_BUDGET_MS`, read by both the watchdog and the gate). A round builds
the candidate and the shipped kernel at the three shapes, so a candidate's build at a timing shape is
limited to a third of what the shipped kernel's three measured times leave of 20 s (about 5.7 s at about
1 s a shape); such a kernel is five times slower than shipped, its speed columns are already 1, and
stopping it loses no ranking. The correctness cases are stopped if projected over a third of the 30 s,
the rungs and pairs if a build would pass the 30 s.

**The estimate.** `SHIPPED_COST_FACTOR` 64 to 640 (2,560 to 25,600 per item): the largest estimate of
any feasible kernel in today's fit_A2 ledgers was 2,499, so the old bound was binding; all eleven kernels it
kept out today (3,370 to 14,713) now run and are measured. The base (20,000) was never binding and is
unchanged. The estimate remains the guard for kernels too big for even the correctness cases, and for
time that scales other than as d*d*m within one 32x step.

**Limits of the model.** The three timing shapes have the same work (2.75e11 item-cells), so one rate
predicts one time for all three; a kernel whose speed differs by shape is predicted by its rate at the
largest pair (d 8192). GPU tests to run when the device is free:
`the_shipped_kernel_predicts_its_own_time_at_the_timing_shapes` (within a factor of two of each measured
time) and `a_deliberately_slow_kernel_is_stopped_before_the_timing_shapes` (64 exact passes of the item
loop: correct, stopped as `slow`, under 30 s).

**Measured 2026-10-11 20:03Z** (release, M3 Max, `examples/wgsl_gram_gate`). Both GPU tests pass. Wall ms at
the three timing shapes, predicted against measured:

| kernel | predicted | measured | rate ms/1e9 | outcome |
|---|---|---|---|---|
| shipped | 1,030 x3 | 1,018 / 1,008 / 1,012 | 3.74 | feasible, HFF 1.186 |
| popA2_g006j g006_r18 (best) | 155 x3 | 106 / 93 / 98 | 0.55 | feasible, HFF 0.103 |
| fit_A2 g004 r3p1 Opus (old estimate 4,552, kept out) | 252 x3 | 210 / 210 / 216 | 0.90 | feasible, HFF 0.360 |
| shipped with 64 item passes | 17,300 at the d 8192 pair | not dispatched | 252 | slow, stopped in 3.9 s |

The limit on this device is 5,654 ms a build. Fast kernels are over-predicted by 1.2-1.6x (the floor and
the pairs' data weigh more for them), the safe direction.

## Expectations, stated before any run

| measurement | expected |
|---|---|
| of 32 generation-0 kernels: parse; validate and compile | 28–32; 24–30 |
| pass the the constraint's 128 test cases; pass all 1,024 and the pair criterion | 10–20; 5–15 |
| correct and faster than shipped on the held-out shapes, generation 0 | 1–5 |
| offspring correct, by operator: rules, model edit, crossover, fresh, repair | 90 %, 40 %, 20 %, 25 %, 30 % |
| best geometric-mean speed-up after 500 evaluations | 2–6× |
| worst-shape speed-up of the champion | ≥ 0.7 × its mean |
| model calls per correct kernel; GPU seconds per correct kernel | 3–6; 60–200 |
| wall clock, steps 0–2 | 3–4 h at the healthy GPU rate (1,024 probes at 100 s a kernel on the slowed GPU is the dominant cost; ten correct kernels) |

## Tasks (each "Read docs/Rules.md, then:")

1. **The caller and the fitness evaluation.** `examples/hgp_gram.rs`: contract prompt + strategies, the CLI call and JSON reply, fence stripping and the header, stages 1–4, the the constraint's 128 test cases (`probes(1)` sampled by seed), ledger and call logs. Tests (in the example's `#[cfg(test)]` or `src/wgsl/hgp.rs` if shared): header parses; fenced reply strips; the shipped text climbs to stage 4; a syntax error stops at stage 1 with its line. Run generation 0; put the stage counts in `docs/PLAN_hgp_gram.md` beside the expectations.
2. **The three mutations and the loop.** Operators A (e-graph through the reader), B, C, fresh, repair; parentage; elites by stage then HFF; 500 evaluations; three seeds if time. Per-operator correctness rates and the champion table into the plan. The share-equivalence scan of the population each generation (reader → `population_dag` fold → homeotic rows by referrers) and its report: distinct subtrees, share rate, the top 20 components, the components correct kernels carry and incorrect ones lack.
2b. **Evolve from components.** The popular components as terminals of the typed sampler and of subtree mutation; the tree search of `wgsl_gram_evolve` rerun with and without them; correct kernels per 100 evaluations, both ways, into the plan.
3. **The champion in Blue World** (`PLAN_wgsl_gram_superopt.md` task 4 unchanged).
4. **The report**: the table against expectations, ≤ 200 words, the research question answered.

## Verification

- `RUSTFLAGS="-D warnings" cargo test --release --features gpu,wgsl` and clippy clean before each commit.
- Generation 0 run announced (what, why, how long, log `logs/HGP/001/run.log`), under nohup with the guard; the log names each candidate before and after it runs.
- A kernel's stage is reproducible from its saved text alone: `wgsl_gram <text>` reproduces its pair numbers; the probes test reproduces its probe count.
