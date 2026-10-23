# v24 ingestion index: the design contract (250,000 records, stages 1 to 5, microbatch keys only)

This problem, `v24_ingest250k`, is `v24_ingest5b` at ten times the records:
250,000 stream rows in five sections of 50,000, and the same 2,500 oracle
queries. The section "v24_ingest250k: the size, and what binding 0 holds"
below states what differs. Everything else is the contract given here in
full: the same bindings, cfg fields and entry points.

## v24_ingest250k: the size, and what binding 0 holds

- **Five sections of 50,000 rows**, 250,000 in all, plus 2,500 queries.
  `ingest` is dispatched with 50,000 workgroups a pass, one a row.
- **During `ingest`, `keys` holds only the current microbatch's keys:**
  50,000 rows (409.6 MB). Row `r` of the buffer is stream row
  `cfg.arrived_begin + r`.
- **During `search`, `keys` holds nothing** (64 zero floats).
- **The arenas are 32 MiB each, as in `v24_ingest5b`.** At 250,000 rows
  that is 268 bytes a row for `index_u32` and `index_f32` together. A full
  copy of every key, even at 8 bits a dimension (2,048 bytes a row), or a
  512-byte code a row, does not fit: an index must keep less a row than an
  exhaustive scan of quantised keys needs.

This is the contract a v24 design is written to. A design is one WGSL module
with two entry points, `ingest` and `search`. fuller's harness grades it
(`problems/v24_ingest_small/problem.toml`, `src/wgsl/harness.rs`) and phylu
evolves it. Designs are not hand-written: models write the first ones, then
evolution varies them and models only repair. The plan is fuller
`docs/PLAN_v24_ingest_evolution.md`. The open design questions are in qdrant
`docs/V24_Evolved_GPU_Streaming_Index.md`, sections 4 and 6.

## The task

Two problems use this contract:

- `v24_ingest` is the evolution target of run 1. It is the full set, judged
  on stages 1 to 3 (15,000 arrived rows, 6 windows). One seed's stage-3 keys
  are 123 MB, under the 128 MiB binding limit. Stages 4 and 5 (164 MB and
  205 MB a seed) wait for keys banded over two bindings, a contract change
  for run 2.
- `v24_ingest_small` is the smoke test (compile, run, judge end to end).
  Today's index misses nothing in 9 of its 10 windows, so its misses input
  has no slope there.

Four seeds (42, 137, 911, 4242) each give every record a 2,048-D f32 key. The
records arrive in five microbatches, called sections, in time order. On the
small set each section is 500 rows: 2,500 rows in all, plus 250 oracle
queries. On the full set each section is 5,000 rows: 25,000 rows plus 2,500
queries.

Each seed has its own index. After each microbatch your `ingest` adds the new
rows to it, and your `search` proposes up to `C_MAX = 64` candidate row ids
per query from it.

The four indexes are independent: each seed has its own arenas and sees only
its own keys. The harness then pools the four seeds' candidate lists and
re-ranks the pool exactly against the raw sparse records (this re-rank is the
harness's job, not yours). It counts the top 10 against exact truth.

Your index decides which rows reach the exact re-rank. Nothing else.

## v24_ingest5b: what binding 0 holds

This section overrides binding 0's row of the table below. In every other
respect the contract is unchanged.

- **During `ingest`, `keys` holds only the current microbatch's keys:**
  5,000 rows (41 MB). Row `r` of the buffer is stream row
  `cfg.arrived_begin + r`, so read row `i` of the microbatch at
  `keys[(i - cfg.arrived_begin) * 2048 ..]`, not `keys[i * 2048 ..]`. No
  earlier row's key is on the device.
- **During `search`, `keys` holds nothing:** it is bound to 64 zero floats.
  `search` sees only `queries`, `index_u32`, `index_f32` and `cfg`.
- **What follows:** a design that needs keys at search time must keep what
  it needs in its own arenas (32 MiB a seed, `index_u32` and `index_f32`)
  while it ingests. For example it can keep quantised codes, projections,
  bucket representatives or centroids. Scanning every key at search time is
  impossible, and no binding exceeds 128 MiB.
- **Scoring:** a candidate is still a stream row id, and the harness still
  re-ranks the pooled candidates exactly against the raw records.
- **What changes elsewhere:** rule 1 under "The rules that matter" holds
  in a stronger form here: only the current microbatch's keys exist. The
  reference kernel's search scans `keys`, so here it finds no candidates. It
  is a measurement reference for the harness only.

## Bindings (group 0, fixed)

| binding | name | access | holds |
|---|---|---|---|
| 0 | `keys` | `storage, read`, f32 (`array<f32>` or `array<vec4<f32>>`) | this seed's keys of the **arrived rows only**: rows `0 .. cfg.arrived_end`, row-major, 2,048 floats a row, in arrival order |
| 1 | `queries` | `storage, read`, f32 | this seed's keys of the queries, `n_queries x 2048` (query q at `q * 2048`) |
| 2 | `index_u32` | `storage, read_write`, u32 (`array<u32>` or `array<atomic<u32>>`) | your index state: buckets, levels, links, counts. A fixed arena of **8,388,608 words (32 MiB)** |
| 3 | `index_f32` | `storage, read_write`, f32 | your f32 state: centres, scores. **8,388,608 floats (32 MiB)** |
| 4 | `cands` | `storage, read_write`, u32 | the search's output: query q's candidates at `q * 64 .. q * 64 + 64`, stream row ids. `0xFFFFFFFF` is a pad. Every slot starts as a pad, so slots you do not write stay pads |
| 5 | `cfg` | `uniform` | `struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32 }` |

`cfg` values:

- `seed_slot`: 0 to 3, for seeds 42, 137, 911, 4242.
- `stage`: 1 to the problem's stages (5 on `v24_ingest5` and `v24_ingest_small`, 3 on `v24_ingest`).
- The microbatch that just arrived is rows `arrived_begin .. arrived_end`, with
  `arrived_begin = (stage - 1) * per_section` and
  `arrived_end = stage * per_section`.
- `pass_index`: 0 to `n_passes - 1` in `ingest`; always 0 in `search`. (The field cannot be named `pass`: it is a reserved word in WGSL.)
- `n_passes`: the N of your `// passes N` header.
- `c_max`: always 64, a contract constant.

Name the uniform variable `cfg` and its fields as above. The safety
contract (`safety.contract` beside this file) gives the legality checker
the fields' ranges by these names, along with 1-D dispatches.

A kernel may leave a binding undeclared. It may not declare one this table
does not list.

## The rules that matter

1. **Only arrived rows exist.** At stage k the `keys` buffer holds rows
   `0 .. k * per_section` and nothing else. No future key is on the device, and
   reading past the end of `keys` reads the binding's LAST float, not zero (the device restricts an out-of-range index to the last element, for loads and stores alike); at search `keys` is 64 zero floats, so every read there is zero.
2. **The arenas live for one run.** `index_u32` and `index_f32` are created
   and zeroed when a run starts. They persist across the run's stages
   and are discarded after it. They are never cleared between stages. Every
   timing run is a whole fresh run, so no measurement reuses a built index.
3. **Each seed's arenas are its own.** Seed slot s never sees another seed's
   state.
4. **A candidate list is a set.** Order does not matter, because the harness
   re-ranks exactly. Duplicates within or across seeds are merged.

   An id that is not an arrived stream row is counted as broken (B). That
   covers a row at or past `arrived_end`, an oracle row, and garbage. Pads
   are ignored.

   Returning more candidates is not free: the pool size P is in the
   fitness, against today's index.

## Entry points and the dispatch program

Each stage runs, for each seed in turn:

1. `ingest` is dispatched `n_passes` times, with `cfg.pass_index` set to 0, 1, ...
   Each dispatch has **`per_section` workgroups**, one per arriving row: the
   row is `arrived_begin + workgroup_id.x`. You choose the workgroup size. A
   pass sees every write of the passes before it; a dispatch boundary is the
   only global barrier.
2. `search` is dispatched once, with **`n_queries` workgroups**, one per
   query: the query is `workgroup_id.x`. You choose the workgroup size.

The harness then reads `cands`. The next stage begins after all four seeds
are done. Each stage is one GPU submission, and it must finish within 4.9 s.

Your module declares `// passes N` on a line of its own, with 1 ≤ N ≤ 16.
The default is 1. A pass loop is how a design builds level by level (count,
then split, then promote, then link) within one microbatch.

Other limits:

- Workgroup memory: 16 KiB.
- At most 256 invocations per workgroup.
- Storage atomics on `index_u32` are allowed: `atomicAdd`, `atomicSub`,
  `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`,
  `atomicExchange`, `atomicLoad` and `atomicStore`. Not
  `atomicCompareExchangeWeak`: naga 0.20's Metal backend has none.
- Use loops bounded by `cfg` values or by constants.
- Every index into an arena should be provably in bounds: mask it or clamp
  it to the arena size. This keeps the design inside what the safe kingdom
  can read (task B6).

## The fitness, in one paragraph

The score is the hff library's HFF TrueNorth angle over a vector of inputs.
Each input is in [0, 1], where 0 is best. Each is mapped by a fixed range
anchored at today's index on the same window, with today's index at 0.5.
Nothing is averaged.

A window is one stage × one oracle half. For each window the inputs are:

- `x_M = min(1, M / (2 M_today))`. M is the top-10 misses, summed over the
  window's queries.
- `x_P = min(1, P / (2 P_today))`. P is the pooled candidates sent to the
  re-rank.
- `x_B = min(1, B / 10)`. B is the broken entries.

A floor of 1 replaces today's count when today's is 0. Misses use today's
counting rule: the re-ranked top 10 are counted against the accept set,
which is every arrived row within 1e-6 of the 10th true cosine, so ties are
found.

There are two more inputs per stage for each fresh timing run (three runs).
The first is the ingest through stage k: your ingest GPU ms for
microbatches 1..k, every pass and seed summed. It is scored against today's
whole build of sections 1..k, because today's index builds from scratch at
every stage. The second is the stage's search GPU ms, four seeds summed,
scored against the sum of today's per-query search ms for the same queries.
Both use `clamp(0.5 - log10(today_ms / kernel_ms) / 4, 0, 1)`.

A design that does not compile or does not run is infeasible.

## The reference kernel

`reference.wgsl` is **not an index design**. It is the harness's measurement
reference: exact brute force over this seed's arrived keys, with `ingest`
doing nothing. It shows the bindings and the entry points in working form.
Do not copy it as a design.

## The design space

Andrew's hypothesis is to index bottom-up. Find hotspots by counting hits.
Splay the data into right-sized buckets. Promote a random sample of the
existing points up through the levels. Build the levels from the bottom, all
of a batch rising together, with a fixed ef per layer and each group's scope
growing level by level. The first fourteen strategies below explore it
directly. The rest cover graphs and beams (15 to 17), hashing and
projections (18 to 22), centres and clustering (23 to 26), and flat and
hybrid designs (27 to 32). Each is a starting direction for one seed
design, not a recipe.

## Strategies

- **Andrew's bottom-up hypothesis, whole.** Count hits to find the hotspots before visiting their rows (pass 0: each new row's coarse cell by the sign bits of fixed projections, atomicAdd counts); splay the data into right-sized buckets (pass 1: a cell over the size threshold split by further sign bits until every bucket holds at most B rows); promote existing points upward at random (pass 2 and on: each bucket's rows promoted to the level above with a fixed probability, the upper levels a random sample drawn from where the data is); a fixed ef per layer; each group's scope growing level by level (a level-l point linked to its nearest promoted points within 2^l buckets); all of a microbatch rising together, level by level, one pass a level. Search descends the levels with that fixed ef from the top, then scans its bottom buckets by what the arenas keep of their rows (codes, not keys: search has none).
- **Hotspot-first splaying.** A count histogram over 4,096 cells finds the
- **Right-sized buckets by density.** Each bucket holds at most B rows,
- **Bottom-up random promotion, fixed ef.** Each new row is promoted to the
- **Level-synchronous build.** One pass per level, bottom first. All the
- **Promotion by bucket centre.** Each bucket's medoid, its most central
- **Several promotions per dense bucket.** A bucket promotes about
- **Growing scope per level.** At level l a point links to the k nearest
- **Upside-down HNSW beside a flat tail.** Batches already settled live in
- **Splay with re-promotion.** When a bucket splits, it re-promotes its new centres and updates the upper links that pointed at it. Old links are kept as second chances.
- **Count-min hotspot sketch.** A small count-min sketch over hashed cells estimates density. Each row is placed at the depth its density calls for.
- **Two-level bucket tree with a fixed fan-out.** There are 64 top cells chosen from the first microbatch's rows. Each cell holds up to 64 child buckets of up to 64 rows. Search probes the top 4 cells and then the top 8 children in each.
- **Bottom-up merge of microbatch trees.** Each microbatch builds its own small hierarchy. A pass then merges its top level into the global top level by nearest-centre links.
- **Rank-preserving buckets.** Rows are bucketed by the rank of their projection onto 16 fixed directions, in quantiles counted with atomics. A bucket is a 16-digit rank code. Search probes nearby codes.
- **Flat k-NN graph with beam search.** Each new row finds its 16 nearest neighbours among the arrived rows by a beam from random entry points, and adds reverse links into fixed-degree slots (replacing the farthest). Search runs a beam of 64.
- **NSW with random long links.** Each row gets 8 near links and 2 random far links. Search runs a greedy beam from 8 random entries per query.
- **Graph over bucket centres.** Buckets are linked by centre similarity. Search beams over the centres, then scans the members of the best buckets.
- **LSH tables.** There are 8 tables of 12-bit sign hashes on independent projections, with fixed-capacity buckets in the arena. Search unions the query's buckets across the tables.
- **Multi-probe LSH.** One table, probing the query's bucket and the buckets that differ in its least-confident bits.
- **Sorted projections.** Rows are kept sorted along 4 fixed projection directions; insertion merges each batch in. Search takes a window around the query's position on each direction and intersects or unions the windows.
- **Cross-polytope hashing.** A bucket is the argmax coordinate (with its sign) of a key projected into 64 dimensions. Search probes the second-best coordinates as well.
- **Product-quantised coarse cells.** Two 32-centroid codebooks over the key's halves, trained on the first batch. A row's cell is its pair of codes. Search probes the best cell pairs.
- **Hierarchical centres (k-means tree).** Centres are taken from the first batch's rows. Each new row is assigned to its nearest centre. A centre that grows past a cap splits in two, using two of its members as the new centres.
- **Online k-means with drift.** There are 256 centres in `index_f32`, updated as running means as rows arrive. Rows are assigned to buckets by centre. Search scans the 8 best centres.
- **Spill assignment.** Each row is written into its 2 nearest buckets, so a query near a border finds it either way.
- **Density-adaptive cell radius.** The first rows found cells. A new row starts a new cell when no cell is within a radius; the radius shrinks where the counts are high.
- **Flat brute force within buckets.** A coarse bucketing, then search scans every row of the query's best 4 buckets exactly in key space and keeps the top 64.
- **Fresh-tail exact scan plus index.** Search always scans the newest microbatch exactly. Older rows come through a bucket index. The 64 slots are split between the two, for example 32 and 32.
- **Seed-aware candidate budget.** The design spends its 64 slots by the seed slot's own confidence (score gaps), filling the rest with pads. Remember that P is in the fitness.
- **Quantised keys.** Each key is quantised to 8 bits per dimension in the arena, for faster bucket scans in search: a 2,048-byte code per row, scored with integer dot products.
- **Score-gap early stop.** Search scans buckets in order of their centre score and stops when the next centre cannot beat the 64th best found.
- **Learned split directions.** Each over-full bucket splits along the principal direction of its members, estimated by a few power iterations in a pass. It splits at the median, which keeps buckets balanced on the Times-Square-shaped data.
