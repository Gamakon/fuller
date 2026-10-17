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
