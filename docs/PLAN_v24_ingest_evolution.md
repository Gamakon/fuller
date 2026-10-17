# PLAN: evolving a bottom-up GPU ingestion index (v24) — the build

## Context

**Objective.** Evolve, in the WGSL kingdom, an ingestion routine and its search: four
seeds' 2,048-D keys stream in, five microbatches in time order, into four independent
per-seed indexes on the GPU; after every microbatch the 2,500 oracle queries search the
four indexes and their pooled candidates go to the exact re-rank. Goal: recall@10 of
0.996+ on first ingestion, as fast as it can be. The design questions (no answers) and the test-set decisions
are in qdrant `docs/V24_Evolved_GPU_Streaming_Index.md` (sections 1–11).

**Why this exists.** Today's index on the same data (qdrant 3eb43d194): recall@10 0.9984,
0.9966, 0.9963, 0.9937, 0.9931 after 5k, 10k, 15k, 20k, 25k records; settle 4.7 s to 31.7 s
for 7.5k to 27.5k rows. Recall falls because collisions rise with more data (Andrew); the
cure is how records are ingested and how many reach the exact re-rank.

**Method.** Nobody writes the index. 32 models (16 Sonnet, 8 Opus, 4 Codex, 4 Fable) write
first designs from one contract; the harness grades them; phylu varies them; models only
repair. The fitness is the agreed HFF (fuller `docs/PLAN_settle_evolution.md` section 2,
commit d1ee890): the hff library's TrueNorth, normalisation off, counts per window mapped
into [0, 1] by fixed ranges anchored at today's count on the same window (today = 0.5),
nothing averaged.

**Status (2026-10-17).** Ready: the test sets (`/Users/andrewmorgan/data/arm/v24_test`,
25,000 + 2,500; `v24_test_small4`, 2,500 + 250), the four seeds' keys
(`db/udv23_keys/v24_main/s<seed>.trlkeys`, TRLKEYS1), exact truth per stage, today's
counts. Not built: everything below.

## 1. The contract a design is written to (the seed prompt's core)

Bindings (group 0), fixed:

| binding | name | kind | holds |
|---|---|---|---|
| 0 | `keys` | storage read, f32 | one seed's keys of every record that will arrive, row-major `n × 2048` (rows in arrival order; a design may read only rows below `cfg.arrived_end`) |
| 1 | `queries` | storage read, f32 | the oracle queries' keys for this seed, `q × 2048` |
| 2 | `index_u32` | storage read_write, u32 | the design's own index state (buckets, levels, links, counts) — an arena of fixed size, zeroed once before the first microbatch, never cleared after |
| 3 | `index_f32` | storage read_write, f32 | the design's own f32 state (centres, scores), same lifetime |
| 4 | `cands` | storage read_write, u32 | the search's output: per query `C_MAX` candidate row ids, `0xFFFFFFFF` pads |
| 5 | `cfg` | uniform | `seed_slot, stage, arrived_begin, arrived_end, n_queries, pass_index, n_passes` (C_MAX is a constant, section 1b; `pass` is a WGSL reserved word) |

Entry points, fixed names: `ingest` (dispatched `n_passes` times per microbatch, `cfg.pass_index`
= 0..n_passes-1; the design declares `// passes N`, 1 ≤ N ≤ 16, and `// grid` its
workgroups per pass from the batch size) and `search` (one dispatch per microbatch after
ingest, one or more threads per query). Atomics on `index_u32` are allowed (load, store, add, sub, min, max, and, or, xor, exchange; NOT atomicCompareExchangeWeak, which naga 0.20's Metal backend does not support). Each seed has
its own `index_u32`/`index_f32`, so the four indexes are independent by construction. The
exact re-rank is not the design's: the harness pools the four seeds' candidates and
re-ranks them exactly against the raw sparse records.

## 1b. Decisions after Codex's review (2026-10-17 00:05Z)

1. **No future keys exist for a design.** At stage k the harness binds a `keys` buffer
   holding only the arrived rows (sections 1..k); nothing later is on the device.
2. **A fixed candidate budget:** `C_MAX = 64` per seed, a contract constant (not a cfg
   field); the pool is the union of the four seeds' lists, deduplicated, at most 256;
   P counts the deduplicated pool. A design cannot buy recall by returning everything.
3. **One counting rule, today's:** after the exact re-rank of the pool, the top 10 are
   counted against the accept set (every arrived record within 1e-6 of the 10th true
   cosine), found capped at 10 — the rule `count_today_index.py` used for today's
   anchors, so M and M_today are the same measure.
4. **Index state lives for one evaluation run:** `index_u32`/`index_f32` are created and
   zeroed at the start of a run, persist across the five stages, and are discarded after;
   every timing repetition is a whole fresh run, so no sample reuses a built index. Timing
   samples are each stage's ingest and search GPU times inside a run (GPU timestamps).
5. **Fitness carries `inputs: Vec<f64>`** beside the old six columns, and θ from the
   library over `inputs` when it is non-empty; phylu reads `inputs` (persisted, compared,
   briefed) for problems that fill it; the Gram and nluca problems are unchanged.
6. **phylu gets a raw-WGSL population path:** a seed the safe reader cannot read yet is
   admitted as raw text, evaluated by the harness (compile, run under the GPU wait
   budget), varied only by model rewrites and repairs until B6 lets it be read; every
   raw seed's reader failure is logged as a kingdom gap for B6.
7. **Anchors per window = stage × half,** recomputed from per-oracle counts (B1 extends
   `count_today_index.py`); `P_today` from the search's `candidates_pooled`.

## 2. The machinery to build (hand-written; designs are not)

| # | where | what | first needed by |
|---|---|---|---|
| B1 | qdrant `experiments/v24_ingest` | `export_harness_case.py`: from a test set, per stage the **accept sets** (for each oracle every arrived record whose exact cosine ≥ the 10th true cosine − 1e-6: ties found, not missed), the raw sparse records (CSR), the keys sliced per seed as raw f32, the oracle split (train / validation by hash, half each), today's anchor counts per stage and half; written as a fuller problem's case files | B3 |
| B2 | fuller harness | **multi-entry dispatch** (`dispatch = { entry = "ingest", ... }`, every entry compiled from the one module), **pass loops from a kernel header** (`// passes N`), **per-stage reads** into an output with a stage axis | B3 |
| B3 | fuller `problems/v24_ingest_small`, `problems/v24_ingest` | the problem: bindings of section 1, setup (keys, queries resident), run program (per stage: per seed `ingest` × passes, `search`, read `cands`), the output kind `pooled_recall` (B4) | B5 |
| B4 | fuller harness | output kind **`pooled_recall`**: per stage and oracle half, pool the four seeds' candidates, re-rank exactly against the raw records, count misses M against the accept sets, count candidates P sent to the re-rank, count broken entries B (ids not yet arrived or out of range); the **HFF inputs** per window (window = stage × half): x = min(1, M/(2 M_today)), min(1, P/(2 P_today)), one input per timing sample (ingest ms and search ms against today's settle and search ms, `clamp(0.5 - log10(s)/4, 0, 1)`), min(1, B/10); θ from the hff library (`score::truenorth_angle`, normalisation off, "truenorth"); Fitness carries a variable-length input vector (today a fixed 6) | B5 |
| B5 | phylu `wgsl_fit` | consume B4's Fitness (variable inputs, θ from the library); the seed contract as the problem's `contract.md` with 32 design strategies; evaluation of a seed's raw WGSL when the safe reader cannot yet read it (generation 0 and model rewrites only, with the GPU wait budget and a shipped-kernel probe after any killed run) | run 1 |
| B6 | fuller kingdom | what designs need to read into the safe kingdom (so crossover and rules apply): several entry points in one module, storage-buffer atomics (`atomicAdd`, `atomicMin/Max`, `atomicCompareExchangeWeak` where naga allows), dynamic indices into `index_u32` proved in bounds by masking, loops bounded by `cfg` values; each gap found by the seeds is a kingdom bug, fixed in the kingdom | run 2 |

## 3. Runs

- **Run 1 (after B1–B5):** 32 seeds on `v24_ingest_small` (2,500 + 250, keys 22.5 MB a seed:
  under the 128 MiB binding limit), the ladder counted (parse, compile, run, recall), then
  20 generations of model-rewrite evolution (the creativity arm B settings) on the seeds
  that run. Reported: θ per generation, recall@10 per stage, ingest and search ms.
- **Run 2 (after B6):** the same with section crossover and rules on designs read into the
  kingdom; then the full 25,000 + 2,500 set once keys are banded over two bindings (225 MB
  a seed exceeds the 128 MiB limit; a contract change, decided then).

## 4. Expectations, stated before any run

| measurement | expected |
|---|---|
| of 32 seeds: compile and run every stage | 12–24 |
| seeds at or above today's recall at every stage (x_M ≤ 0.5) | 0–5 |
| best recall@10 at stage 5 after 20 generations, small set | 0.993–0.999 |
| designs read into the kingdom before B6 | 0–4 |

## 5. Tasks and who builds them

B1 a subagent in qdrant; B2+B3+B4 one subagent in fuller (one owner of the harness); B6 a
subagent in fuller's kingdom files (safe_reader, render, check, gen_tables — not the
harness); B5 after B4 lands. This session reviews, runs every GPU job, and launches runs.
One GPU process at a time; CPU tests only in subagents.

## 6. The operators' gate on two-entry modules (strict layer 7, measured on the GPU 2026-10-17 11:44Z)

This replaces the CPU measurement of 09:48Z (fuller 29d7761): the value and race stages
now run on the device.

**Result.** Under strict layer 7, 7 to 13 % of each operator's children pass every stage,
and one child of 1,000 races. Layer 7 is still the wall: it turns away 23 of the 26 seeds
that read, and the children that pass descend almost all from the three it accepts (seeds
24, 25, 26). Every child that reached the value check computed its own text's values.

**How.** phylu `wgsl_fit --measure-operators 200 --measure-races 3 --seed 1` on
`problems/v24_ingest_small` (phylu 606c5843, fuller 69039be, clean worktrees; 60.8 s, of
which 6.1 s on the device). Parents: the seeds of `logs/V24/run1/seeds` that read into the
kingdom at head 32 and 160 tail slots, each under its own role (26 of 32, under 10 roles;
three more need 179 or fewer slots, so a 184-slot layout reads them too). 200 children per
operator. On the host, in order: **role** (the receiving parent's, plus the donor's rows
its tree uses, phylu `wgsl_model::child_role`), **render** under that role, **naga**,
**safety** (`knob::check_legal_under`, layers 1 to 7 under `safety.contract`, strict), **read
back** into the fixed table and decoded. On the device (fuller's harness, one run at a
time): **values** (the child's text and the text its genome expresses give the same
outputs on `cases/admission.toml`; byte-equal texts need no run) and **no race** (three
fresh runs, each with zeroed arenas: a race is a query's pooled candidate set, the four
seeds' lists with pads dropped, that differs; candidates that differ only in order, and
arena state that differs with the outputs the same, are counted and are not failures).
Records: `logs/V24/gate/strict/gate.md`, `gate.tsv`.

| operator | children | made | role | render | naga | safety | read back | values | no race (3 runs) | of which order-only | of which state-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (parents) | 26 | 26 | 26 | 26 | 26 | 3 | 3 | 3 | 3 | 0 | 1 |
| section crossover | 200 | 200 | 197 | 197 | 197 | 18 | 18 | 18 | 18 | 0 | 5 |
| typed crossover | 200 | 200 | 192 | 192 | 191 | 19 | 19 | 19 | 19 | 0 | 10 |
| rule | 200 | 200 | 200 | 200 | 200 | 18 | 18 | 18 | 18 | 0 | 7 |
| generic rewrite | 200 | 157 | 157 | 157 | 156 | 15 | 15 | 15 | 14 | 0 | 2 |
| point mutation | 200 | 200 | 194 | 194 | 193 | 26 | 26 | 26 | 26 | 1 | 9 |

Every count above rests on fuller 69039be, committed, with no uncommitted edits (the
lead's clean worktrees). The re-run at 12:11Z (phylu 1790a82e, the same fuller, layout
152 slots from the seeds' need) gives the same table but section crossover: 17 with no
race of 18 (one race seen in 3 runs that the 11:44Z runs did not show). A race shows in
some runs only, so 3 runs under-count races; the arena measurement runs 10.

Of the 99 parents and children that reached the race check, 86 express their text
byte-equal and 12 were run against it on the device; none computed different values. The
one race: rewrite child 23 of seed 26, whose third run's pooled candidates for one
stage-query held 35 rows against the first's 34.

**Failures, named** (children, with the seeds that fail the same way in parentheses).

| failure | count | owner | status |
|---|---:|---|---|
| layer 7 `output_read_of_another_invocations_cell`: a read of `index_f32` another invocation writes | 291 (8 seeds) | kingdom (check.rs) | open: index state read across invocations and passes; the arena mode (v24-b6-kingdom) is the answer under test |
| layer 7 `write_sets_overlap`: two stores of different index forms, or a data-dependent index, may write one cell of `index_f32` / `index_u32` | 261 (7) | kingdom | open: arena mode |
| layer 7 `store_index_not_affine`: a store to `cands` or `index_f32` at an index not affine in the thread's coordinates | 179 (5) | kingdom | open: arena mode |
| layer 7 `workgroup_memory_race` | 102 (3) | kingdom | open: some may be real races in the seeds |
| layer 7 `subgroup_operation_in_nonuniform_control` | 8 | operators (rewrites) | the child is illegal, as it should be |
| the parents' roles conflict: a donated row needs another element view of the same binding (`array<f32>` and `array<vec4<f32>>` keys or queries; `array<u32>` and `array<atomic<u32>>` index) | 17 | phylu `child_role` | named, never skipped; the same binding under another name is renamed (phylu c6dcc6d7) |
| the rendering is not WGSL: `vec4<f32>([...], ...)`: kingdom rows `load.phi_safe`, `load.phi4_safe`, `load.h_safe`, `load.phi_len`, `load.h_len` render with the role's input and output, which a v24 role does not name, and the table offers them to point mutation | 2 | kingdom (render.rs) | open, with v24-b6-kingdom |
| workgroup memory over the device's 32,768 bytes | 1 | the child | illegal, as it should be |
| generic rewrite with no site in the parent | 43 | operator | expected |
| `kernel.entry_safe is not a grid row` (`params::resolve` on a module root) | 0 | kingdom | fixed, fuller cf6c9cd |

**What each operator needs next.** None of the operators is the bottleneck: each makes
children whose roles, renderings and read-backs hold. Layer 7 is: the v24 arena
(`index_u32`, `index_f32`, `cands`) is written at data-dependent indexes and read across
invocations, which the ownership model (each invocation owns cells at affine indexes)
cannot prove. The decision is the kingdom's: either an arena contract (atomics-only
access; reads across passes as ordered by the dispatch program), or designs restricted to
affine ownership.

**Seeds and layout.** Run 1's readable seeds are trees under their own roles (phylu
3baeafa2). At fuller 69039be (committed), 26 of the 32 read; the gate's layout from their
largest need, rounded up to eight, is 152 tail slots, and run 2's 128 cannot hold them
all. The counts of 18 (08:43Z), 25 (09:43Z) and 28 seeds needing 179 slots (11:00Z) were
measured with v24-b6-kingdom's uncommitted fuller edits in the tree, not at a commit.

### The strict arm (the baseline the arena arm is compared with)

**Result.** Under strict layer 7, with every readable seed in the layout and 10 race runs,
7.5 to 11.5 % of each operator's children pass every stage and none races.

Run on the GPU 12:37–12:38Z: fuller 1fbdcbc (committed, no uncommitted edits in src/ or
kingdoms/), phylu b68c08f3, `--measure-operators 200 --measure-races 10 --seed 1` on
`problems/v24_ingest_small`, layout head 32 with 184 tail slots from the seeds' need; 28
parents of 32 seeds under 10 roles; 86.3 s, of which 29.7 s on the device. Records:
`logs/V24/gate/strict/gate.md`, `gate.tsv`.

| operator | children | made | role | render | naga | safety | read back | values | no race (10 runs) | of which order-only | of which state-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (parents) | 28 | 28 | 28 | 28 | 28 | 3 | 3 | 3 | 3 | 0 | 1 |
| section crossover | 200 | 200 | 198 | 198 | 198 | 23 | 23 | 23 | 23 | 0 | 7 |
| typed crossover | 200 | 200 | 192 | 192 | 192 | 21 | 21 | 21 | 21 | 0 | 5 |
| rule | 200 | 200 | 200 | 200 | 200 | 18 | 18 | 18 | 18 | 0 | 6 |
| generic rewrite | 200 | 168 | 168 | 168 | 168 | 15 | 15 | 15 | 15 | 0 | 6 |
| point mutation | 200 | 200 | 198 | 198 | 197 | 23 | 23 | 23 | 23 | 0 | 10 |

Of the 103 parents and children through every stage, 10 were run against their expressed
text on the device and none computed different values; in 10 fresh runs none raced and
none differed in candidate order; 1 parent and 34 children differed in arena state alone.
The safety layers turn away 25 of the 28 seeds: layer 7 24 (write_sets_overlap 8,
output_read_of_another_invocations 8, store_index_not_affine 5, workgroup_memory_race 3),
layer 4 one (a barrier under a condition that is not workgroup-uniform).
