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
| 3 | 128 admission probes (a fixed, seeded sample across families A–E); then the full 1,024 and the fixed pairs for the top 8 of the population | probes passed; pair criterion max ratio |
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
| The 128-probe admission is unstratified and can omit every band-forcing probe (Codex 2); probes are too small for tiled kernels (Claude 4.6); the probe pass is a mean over cells and lets one cell be 1 % wrong (Claude 4.7); stage 3's ranking mixes denominators (Codex 3). | Admission is stratified across families with every band-forcing D probe included; probes added at d ≥ 130, m ≥ 2 × the largest tile; pass is the per-cell maximum; a kernel is "correct" only after all 1,024 probes and all training pairs; stage 4 requires it. |
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
| converted kernels compiled and run on the device (feasible under the 128 admission cases) | 25–27 | not run: the GPU is held at 100 % by the kernel orphaned on 2026-10-10 (`ioreg` Device Utilization 100 % with no process of ours); needs the machine restart |

The dumps are `logs/HGP/001/safe/karva/kNN.json`; the converted kernels `logs/HGP/001/safe/kNN.safe.wgsl`. The run that follows the restart: `wgsl_fit --kingdom safe --dumps fuller/logs/HGP/001/safe/karva` (phylu), generation 0 first, every row evaluated once; expected: the 27 feasible, HFF as generation 0 measured them.

## Expectations, stated before any run

| measurement | expected |
|---|---|
| of 32 generation-0 kernels: parse; validate and compile | 28–32; 24–30 |
| pass the 128 admission probes; pass all 1,024 and the pair criterion | 10–20; 5–15 |
| correct and faster than shipped on the held-out shapes, generation 0 | 1–5 |
| offspring correct, by operator: rules, model edit, crossover, fresh, repair | 90 %, 40 %, 20 %, 25 %, 30 % |
| best geometric-mean speed-up after 500 evaluations | 2–6× |
| worst-shape speed-up of the champion | ≥ 0.7 × its mean |
| model calls per correct kernel; GPU seconds per correct kernel | 3–6; 60–200 |
| wall clock, steps 0–2 | 3–4 h at the healthy GPU rate (1,024 probes at 100 s a kernel on the slowed GPU is the dominant cost; ten correct kernels) |

## Tasks (each "Read docs/Rules.md, then:")

1. **The caller and the fitness evaluation.** `examples/hgp_gram.rs`: contract prompt + strategies, the CLI call and JSON reply, fence stripping and the header, stages 1–4, the 128 admission probes (`probes(1)` sampled by seed), ledger and call logs. Tests (in the example's `#[cfg(test)]` or `src/wgsl/hgp.rs` if shared): header parses; fenced reply strips; the shipped text climbs to stage 4; a syntax error stops at stage 1 with its line. Run generation 0; put the stage counts in `docs/PLAN_hgp_gram.md` beside the expectations.
2. **The three mutations and the loop.** Operators A (e-graph through the reader), B, C, fresh, repair; parentage; elites by stage then HFF; 500 evaluations; three seeds if time. Per-operator correctness rates and the champion table into the plan. The share-equivalence scan of the population each generation (reader → `population_dag` fold → homeotic rows by referrers) and its report: distinct subtrees, share rate, the top 20 components, the components correct kernels carry and incorrect ones lack.
2b. **Evolve from components.** The popular components as terminals of the typed sampler and of subtree mutation; the tree search of `wgsl_gram_evolve` rerun with and without them; correct kernels per 100 evaluations, both ways, into the plan.
3. **The champion in Blue World** (`PLAN_wgsl_gram_superopt.md` task 4 unchanged).
4. **The report**: the table against expectations, ≤ 200 words, the research question answered.

## Verification

- `RUSTFLAGS="-D warnings" cargo test --release --features gpu,wgsl` and clippy clean before each commit.
- Generation 0 run announced (what, why, how long, log `logs/HGP/001/run.log`), under nohup with the guard; the log names each candidate before and after it runs.
- A kernel's stage is reproducible from its saved text alone: `wgsl_gram <text>` reproduces its pair numbers; the probes test reproduces its probe count.
