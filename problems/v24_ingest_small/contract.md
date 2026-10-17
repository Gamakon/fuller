# v24 ingestion index: the design contract

This is the contract a v24 design is written to. A design is one WGSL module
with two entry points, `ingest` and `search`. fuller's harness grades it
(`problems/v24_ingest_small/problem.toml`, `src/wgsl/harness.rs`) and phylu
evolves it. Designs are not hand-written: models write the first ones, then
evolution varies them and models only repair. The plan is fuller
`docs/PLAN_v24_ingest_evolution.md`. The open design questions are in qdrant
`docs/V24_Evolved_GPU_Streaming_Index.md`, sections 4 and 6.

## The task

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
- `stage`: 1 to 5.
- The microbatch that just arrived is rows `arrived_begin .. arrived_end`, with
  `arrived_begin = (stage - 1) * per_section` and
  `arrived_end = stage * per_section`.
- `pass_index`: 0 to `n_passes - 1` in `ingest`; always 0 in `search`. (The field cannot be named `pass`: it is a reserved word in WGSL.)
- `n_passes`: the N of your `// passes N` header.
- `c_max`: always 64, a contract constant.

A kernel may leave a binding undeclared. It may not declare one this table
does not list.

## The rules that matter

1. **Only arrived rows exist.** At stage k the `keys` buffer holds rows
   `0 .. k * per_section` and nothing else. No future key is on the device, and
   reading past `arrived_end` reads zero.
2. **The arenas live for one run.** `index_u32` and `index_f32` are created
   and zeroed when a run starts. They persist across the run's five stages
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

There is one more input per timing sample. Each fresh run gives each stage's
ingest GPU ms (all its passes and seeds, summed) and search GPU ms (all four
seeds, summed). Each is scored against today's ms (today's settle of the
stage, today's search of the same queries) as
`clamp(0.5 - log10(today_ms / kernel_ms) / 4, 0, 1)`.

A design that does not compile or does not run is infeasible.

## The reference kernel

`reference.wgsl` is **not an index design**. It is the harness's measurement
reference: exact brute force over this seed's arrived keys, with `ingest`
doing nothing. It shows the bindings and the entry points in working form.
Do not copy it as a design.

## 32 design strategies (one per seed design; each is a starting direction, not a recipe)

Andrew's hypothesis is to index bottom-up. Find hotspots by counting hits.
Splay the data into right-sized buckets. Promote a random sample of the
existing points up through the levels. Build the levels from the bottom, all
of a batch rising together, with a fixed ef per layer and each group's scope
growing level by level. Strategies 1 to 14 explore it directly.

1. **Count-then-splay buckets.** Pass 0 hashes each new row to a coarse cell
   (the sign bits of 8 fixed projections of the key) and counts its hits with
   `atomicAdd`. Pass 1 splits every cell over a size threshold into
   sub-buckets on further sign bits. Pass 2 appends the rows. Search probes the
   query's cell and its 1-bit neighbours.
2. **Hotspot-first splaying.** A count histogram over 4,096 cells finds the
   dense cells before any row is placed. Only those cells get extra split
   levels; sparse cells stay flat. Search goes deeper only where the data is
   dense.
3. **Right-sized buckets by density.** Each bucket holds at most B rows,
   where B is a constant such as 32. An over-full bucket splits at the median
   of its widest projected dimension, so the bucket count follows density.
4. **Bottom-up random promotion, fixed ef.** Each new row is promoted to the
   level above with probability 1/16, then 1/256, and so on. Each level links
   a point to its nearest promoted points among the batch and the level's
   existing points. Search descends greedily with ef 32 on every level.
5. **Level-synchronous build.** One pass per level, bottom first. All the
   batch's rows at a level find their links together. No invocation reads
   another's writes within a pass, which makes the build order-independent.
6. **Promotion by bucket centre.** Each bucket's medoid, its most central
   member by key dot product, is promoted. The upper level is a sample drawn
   from where the data actually is.
7. **Several promotions per dense bucket.** A bucket promotes about
   √(bucket size) members, so dense regions get more upper-level points.
8. **Growing scope per level.** At level l a point links to the k nearest
   points within a window of 2^l buckets. Search widens its scope as it
   climbs down.
9. **Upside-down HNSW beside a flat tail.** Batches already settled live in
   a bottom-up graph. The microbatch that just arrived is scanned exactly in
   search, so it is never missed.
10. **Splay with re-promotion.** When a bucket splits, it re-promotes its new
    centres and updates the upper links that pointed at it. Old links are kept
    as second chances.
11. **Count-min hotspot sketch.** A small count-min sketch over hashed cells
    estimates density. Each row is placed at the depth its density calls for.
12. **Two-level bucket tree with a fixed fan-out.** There are 64 top cells
    chosen from the first microbatch's rows. Each cell holds up to 64 child
    buckets of up to 64 rows. Search probes the top 4 cells and then the top 8
    children in each.
13. **Bottom-up merge of microbatch trees.** Each microbatch builds its own
    small hierarchy. A pass then merges its top level into the global top
    level by nearest-centre links.
14. **Rank-preserving buckets.** Rows are bucketed by the rank of their
    projection onto 16 fixed directions, in quantiles counted with atomics. A
    bucket is a 16-digit rank code. Search probes nearby codes.

Graphs and beams:

15. **Flat k-NN graph with beam search.** Each new row finds its 16 nearest
    neighbours among the arrived rows by a beam from random entry points, and
    adds reverse links into fixed-degree slots (replacing the farthest). Search
    runs a beam of 64.
16. **NSW with random long links.** Each row gets 8 near links and 2 random
    far links. Search runs a greedy beam from 8 random entries per query.
17. **Graph over bucket centres.** Buckets are linked by centre similarity.
    Search beams over the centres, then scans the members of the best
    buckets.

Hashing and projections:

18. **LSH tables.** There are 8 tables of 12-bit sign hashes on independent
    projections, with fixed-capacity buckets in the arena. Search unions the
    query's buckets across the tables.
19. **Multi-probe LSH.** One table, probing the query's bucket and the
    buckets that differ in its least-confident bits.
20. **Sorted projections.** Rows are kept sorted along 4 fixed projection
    directions; insertion merges each batch in. Search takes a window around
    the query's position on each direction and intersects or unions the
    windows.
21. **Cross-polytope hashing.** A bucket is the argmax coordinate (with its
    sign) of a key projected into 64 dimensions. Search probes the
    second-best coordinates as well.
22. **Product-quantised coarse cells.** Two 32-centroid codebooks over the
    key's halves, trained on the first batch. A row's cell is its pair of
    codes. Search probes the best cell pairs.

Centres and clustering:

23. **Hierarchical centres (k-means tree).** Centres are taken from the first
    batch's rows. Each new row is assigned to its nearest centre. A centre
    that grows past a cap splits in two, using two of its members as the new
    centres.
24. **Online k-means with drift.** There are 256 centres in `index_f32`,
    updated as running means as rows arrive. Rows are assigned to buckets by
    centre. Search scans the 8 best centres.
25. **Spill assignment.** Each row is written into its 2 nearest buckets, so
    a query near a border finds it either way.
26. **Density-adaptive cell radius.** The first rows found cells. A new row
    starts a new cell when no cell is within a radius; the radius shrinks
    where the counts are high.

Flat and hybrid:

27. **Flat brute force within buckets.** A coarse bucketing, then search
    scans every row of the query's best 4 buckets exactly in key space and
    keeps the top 64.
28. **Fresh-tail exact scan plus index.** Search always scans the newest
    microbatch exactly. Older rows come through a bucket index. The 64 slots
    are split between the two, for example 32 and 32.
29. **Seed-aware candidate budget.** The design spends its 64 slots by the
    seed slot's own confidence (score gaps), filling the rest with pads.
    Remember that P is in the fitness.
30. **Quantised keys.** Each key is quantised to 8 bits per dimension in the
    arena, for faster bucket scans in search: a 2,048-byte code per row,
    scored with integer dot products.
31. **Score-gap early stop.** Search scans buckets in order of their centre
    score and stops when the next centre cannot beat the 64th best found.
32. **Learned split directions.** Each over-full bucket splits along the
    principal direction of its members, estimated by a few power iterations
    in a pass. It splits at the median, which keeps buckets balanced on the
    Times-Square-shaped data.
