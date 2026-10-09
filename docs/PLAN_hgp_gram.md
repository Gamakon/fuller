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

**Approach.** Staged fitness: a candidate climbs rungs (parses → validates and
compiles → correct on probes and pairs → fast), selection by highest rung then
the HFF angle over the performance columns within the top rung; invalid
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

## Design

### The ladder

| rung | test | measured as |
|---|---|---|
| 1 | `naga::front::wgsl::parse_str` | pass, or the first error line |
| 2 | naga validation; Metal pipeline (`Gram::compile_with` through an explicit layout) | pass, or the first error |
| 3 | 128 admission probes (a fixed, seeded sample across families A–E); then the full 1,024 and the fixed pairs for the top 8 of the population | probes passed; pair criterion max ratio |
| 4 | paired timing against shipped on 3 held-out shapes, 3 rounds, random order | speed-up per shape, geometric mean, worst shape, spread |

Within rung 4 the HFF angle over: speed-up on each shape (reciprocal, shipped = 1),
worst-shape speed-up, spread over rounds; all against the shipped kernel. Within rung 3,
probes passed. A rung-4 champion is never displaced by a lower-rung candidate.

### The model calls (`hgp::llm` in the example)

- Command: `claude -p --model sonnet --output-format json --no-session-persistence --system-prompt <contract>` with the task on stdin; the JSON's `result` field is the reply; fences stripped; the kernel must start with a fixed header comment `// grid x y bx by` that the harness parses into `render::Grid` (the dispatch rule `ceil(d/(x·bx)) × ceil(r_rows/(y·by))`).
- Contract (system prompt): the three bindings and `Cfg` verbatim, item-major Φ, band semantics (`r_rows`, `r_off`, read-add-write into `h`), the dispatch rule, the shipped kernel as reference, "return only the kernel".
- Diversity: a strategy per call from a fixed list of 10 (baseline, register block 2×2, 4×4, workgroup tile of items, vector loads of four columns, symmetric half, split accumulators, unrolled item loop, two combined, unlike the shipped kernel) plus a seed phrase; each prompt and reply saved to `logs/HGP/<run>/calls/<id>.{prompt,reply}.txt` with parentage.
- Mutation B (model edit): the kernel, its rung and measurements, one instruction from {reduce memory traffic, change the accumulation, tile differently, vectorise, drop the symmetric half's work, fix this error: <diagnostic>}.
- Mutation C (crossover): two parents' texts and measurements, "combine the named strategy of each"; typed subtree crossover in tree form when both parents read.
- Repair: one per invalid candidate, the exact diagnostic quoted.

### Mutation A: the e-graph editor (Andrew: "a strong editor given a starting program")

`read(text)` → per function `chromosome_with_roots` → `mutants_of_roots(roots, min_ops=2, k=8, iters=4, with_loads=true)` → each mutant's roots through `rebuild_folded` → text → rung 1. Also the `--variants` share subsets (`homeotic::unfold_slot`) as structural variants. Probability 0.35 at the start; a model edit 0.35, crossover 0.15, fresh 0.10, repair 0.05; the rates measured per operator and reported (adaptive weights later, not in 001).

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

Population 32; generation 0 = 32 fresh calls; later generations keep up to 8 rung-4 elites
and make 24 offspring by the operator probabilities; 500 evaluations (~16 generations);
ledger row per candidate (gen, id, operator, parents, rung, diagnostic, probes passed, pair
ratio, speed-ups, angle, path to text); the run restartable from the ledger; the cost bound
and watchdog of `wgsl_gram_evolve` reused; one GPU process; the GPU's speed re-measured
first (shipped build 2.35 s at bench, 15:43; 8.4 s at 19:14 — a machine restart is needed
before rung-4 timing means anything; rungs 1–3 can run regardless).

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
| wall clock | 4–6 h |

## Tasks (each "Read docs/Rules.md, then:")

1. **The caller and the ladder.** `examples/hgp_gram.rs`: contract prompt + strategies, the CLI call and JSON reply, fence stripping and the header, rungs 1–4, the 128 admission probes (`probes(1)` sampled by seed), ledger and call logs. Tests (in the example's `#[cfg(test)]` or `src/wgsl/hgp.rs` if shared): header parses; fenced reply strips; the shipped text climbs to rung 4; a syntax error stops at rung 1 with its line. Run generation 0; put the rung counts in `docs/PLAN_hgp_gram.md` beside the expectations.
2. **The three mutations and the loop.** Operators A (e-graph through the reader), B, C, fresh, repair; parentage; elites by rung then HFF; 500 evaluations; three seeds if time. Per-operator correctness rates and the champion table into the plan. The share-equivalence scan of the population each generation (reader → `population_dag` fold → homeotic rows by referrers) and its report: distinct subtrees, share rate, the top 20 components, the components correct kernels carry and incorrect ones lack.
2b. **Evolve from components.** The popular components as terminals of the typed sampler and of subtree mutation; the tree search of `wgsl_gram_evolve` rerun with and without them; correct kernels per 100 evaluations, both ways, into the plan.
3. **The champion in Blue World** (`PLAN_wgsl_gram_superopt.md` task 4 unchanged).
4. **The report**: the table against expectations, ≤ 200 words, the research question answered.

## Verification

- `RUSTFLAGS="-D warnings" cargo test --release --features gpu,wgsl` and clippy clean before each commit.
- Generation 0 run announced (what, why, how long, log `logs/HGP/001/run.log`), under nohup with the guard; the log names each candidate before and after it runs.
- A kernel's rung is reproducible from its saved text alone: `wgsl_gram <text>` reproduces its pair numbers; the probes test reproduces its probe count.
