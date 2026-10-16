# PLAN: evolving a GPU settle for the trellis — experiment S1

## Context

**Objective.** A wgpu (WGSL) version of trellis's CPU settle, built by evolution:
24 Opus kernels written from the CPU code, then evolved on correctness and speed,
as the nluca and k-means kernels were, until a GPU settle exists that matches the
CPU's quality and beats its wall time.

**Approach.** The settle of one window is a parallel phase (every row of the window
proposes its edges against the state frozen at the window's start: a walk, a beam,
a prune) and a serial phase (seat, commit, advance). The parallel phase is the
compute: it runs once per row at arrival and again in the rolling correction, which
the code measured at 51 % of the settle wall. Problem 1 is that phase, `trellis_propose`.
Problem 2 (seat, commit, advance) follows once problem 1 has a correct kernel.

**Unique feature.** Nobody hand-writes the GPU kernel. The model writes 24 different
first attempts from the CPU source; the harness judges them on recorded CPU output;
phylu's crossover, rules and rewrites make the variation; the model only repairs.

**Status (2026-10-16).** Read in full: `SettleStream::ingest_window`, `walk`,
`beam_over_edges_impl`, `spread_prune`, `seat_record`, `plant_leaf`,
`commit_edges_batch`, `advance_cell_window`, `rand_exits`, `slot_position`,
`hash_below`, `correction_pass_inner` (qdrant `lib/trellis/src/template.rs`) and
`streaming_posture` (`ultradimdb/src/template_ops.rs:216`). Nothing is built.

**Why this order.** Proposals are independent per row (the order-independence the
code guarantees), so they map to GPU threads or workgroups directly; the serial phase
is cheap bookkeeping once deferred re-prune is on. The low-level HNSW GPU shaders
(inventory A5) are not used anywhere here: they failed recall tests (Andrew,
2026-10-16).

## 1. Problem 1: `trellis_propose`

For each row `r` of a window, against the frozen state (the production posture,
`streaming_posture`: walk ef 8, 48 expansions, row budget 64, frozen slots not
skipped, cell cap 0; beam ef 64, 128 expansions; m 13, m_max 16; the floor the
server computes, `noise_floor::exact_null_floor(d_proj, m_max)`
(`template_ops.rs:335`); master disk on, `template_ops.rs:198`):

1. `walk(tpl, key[r], seed, 0, Some(seating, keys), route)`: best-first over the
   wiring (9 proximity exits from the table, a slot taking its cell's; 12 random
   exits from the `mix64` hash, a slot dealing its own), claimed slots scored by their holder's key, unclaimed slots
   materialised on the rotated disk (`centre + sigma * (g - (g.c) c)`, normalised),
   the first visit to a cell collecting every leaf row of the cell, scored;
   out: best cells (descending) and the 64 best-scoring rows.
2. `beam_over_edges(tier, keys, key[r], walk rows minus r, 64, 128)`: best-first over
   the record edges, every scored row returned, descending.
3. Drop `r` and sub-floor rows; `spread_prune` to 13 (keep a candidate only if its
   similarity to `r` exceeds its dot with every kept one; fill from the nearest).

**Outputs per row**: the chosen ids and similarities (13 each, a count), the routed
cell and the next two cells (seat and the two extra homes consume them).

**Inputs, flattened for the GPU** (capture writes them; sizes at the reference shape):
field positions l1/l2/l3 (135 + 2,795 + 55,900 = 58,830 x 128 f32, 28.7 MiB), the proximity exits
(upper_n x 9 u32), the scoping lists for `rand_exits` (`l1_l3_desc`, `l2_l3_children`
as CSR), the master disk (2,000 x 128 f32), sigma, the seed; the seating as CSR
(holder per claimed slot, `cell_slots`, `cell_leaf_slots`, leaves per slot); the
record edges at fixed stride (n x 16 ids and similarities, a count); the keys
(n x 128 f32, 24.4 MiB); the window's rows. Every list keeps the CPU's insertion
order (`cell_slots`, `cell_leaf_slots`, each slot's leaves, each edge list): the walk
collects first-seen in that order, so a CSR in another order is another state.

**The contract states exactly** (Codex review, 2026-10-16): `TEMPLATE_DOMAIN`, u64
`mix64` and its rotates, `hash_below`'s masked rejection (never a modulo),
`rand_exits`'s three distinct L1 draws then the scoped L2, L3 and L4 draws with their
fallbacks; `Cand`'s order (similarity, then the larger id) in both heaps, the ef stop
on a strict `<`, the final sorts by similarity alone; the row removed from the walk's
rows before the beam and again after it (an existing edge can lead back to it); the
claimed slot scored by its holder's key, the unclaimed slot materialised; the cell's
claimed slots pushed and its leaves gathered on its first visit only.

**The reference shape** (`ShapeSpec::production_geometry(50_000, 128)`, the code's own
"≤10-minute reference shape"): production field counts 135 / 2,795 / 55,900, planted
keys, 1,000-row windows, settle seed 42. Cases: windows 5, 25 and 49 (5,000, 25,000
and 49,000 rows arrived: a thin, a middling and a mature index); timing on the same.

## 2. The fitness

- **Validity, every row, a constraint**: ids in the arrived set and not `r`, distinct,
  similarity within 2·d·eps of the f64 dot of the two keys (the CPU's own f32 dot, four
  lanes and a tail, is), at or above the floor, at most 13; the routed cell in range.
- **The routed cell, a constraint**: equal to the CPU's for at least 0.95 of the rows
  (bar set before measuring, like recall's). The route decides the seating, and a
  kernel that skips the walk (brute-force nearest neighbours, a constant route) fails
  here; the walk's 64 rows are also an output, their recall against the CPU's walk
  rows reported (a column, not a gate, until T4's data sets one).
- **Held-out cases**: a second settle seed (43) at the same windows, never used for
  selection, judges the champion only (no case specialisation).
- **Agreement with the CPU, the slope**: per row, the share of the CPU's chosen ids the
  kernel chose; mean over the window's rows; and the routed cell's agreement. The g9
  precedent: GPU rounding flips near ties and reroutes a search, so bit identity is
  not the bar; a kernel passes at mean recall ≥ 0.95 of the CPU's lists (stated before
  measuring; S1-T4 reports the distribution so the bar can be set from data).
- **Speed** (M2 adds it; the harness's reference today is a WGSL kernel timed beside
  the candidate): the kernel's GPU time for the window against the CPU's measured wall for
  the same proposals on this machine (rayon, all cores, median of 5), recorded in the
  case. There is no shipped GPU kernel, so the CPU time is the reference.
- **The end-to-end gate (S1-T7)**: the best kernel's lists fed through the CPU's seat
  and commit over a whole stream; serving recall@10 of that index against the
  CPU-settled index on the same stream, at most 0.002 below it (the g9 rule).

## 3. Seeding: 24 Opus kernels

`wgsl_fit --seed-from model --seed-model 24` with the seeding model named Opus (today
fixed to Sonnet in phylu `examples/wgsl_fit.rs:2384`). Each call gets the contract
(task, bindings, layouts, the CPU code verbatim) and one of 24 design strategies, plus
a seed phrase and "be unlike the obvious kernel". The `claude` CLI has no temperature
setting, so the spread comes from the strategies and the similarity filter (a seed
whose shared sections with an accepted seed cover 80 % of the smaller is replaced).
Strategies include: a thread per row with private heaps; a workgroup per row with
workgroup heaps and a parallel frontier; a lock-step round of all rows (the g9
lockstep form); a bitonic sort for the prune; a two-kernel-free single pass with the
beam's frontier in workgroup memory; subgroup reductions for the dots; a bounded
fixed-iteration beam; walk and beam fused or split by phase; dots vectorised by vec4.

## 4. Evolution

The creativity runs' settings (`logs/HGP/001/queue_creativity.sh`): intake 18, champion
8, 20 generations, every rule, rewrites 0.5, section crossover, model rewrites 0.1 with
repairs (arm B, the fastest on both its seeds), three seeds. One GPU process at a time.

## 5. Machinery (hand-written; the kernels are not)

| # | where | what | needed by |
|---|---|---|---|
| M1 | qdrant `ultradimdb/examples/settle_capture.rs` (approval: Andrew; there both `trellis` and `noise_floor` are reachable) | runs the reference shape through `SettleStream` in the production posture; at windows 5, 25, 49 writes the frozen state flattened, the window's rows, the CPU's outputs and its timed wall; no library change | T2 |
| M2 | fuller harness | an output kind `neighbours` (validity and recall above) and a speed reference from the case (`reference_ms`) when the problem has no shipped kernel | T2 |
| M3 | fuller `problems/trellis_propose` | `problem.toml`, `contract.md` (with the CPU code), `safety.contract` (dispatch bounds), the cases | T4 |
| M4 | phylu `wgsl_fit` | the seeding model as a flag (`--seed-model-name opus`) | T4 |
| M5 | fuller kingdom | whatever the 24 seeds show the safe reader lacks (each one our bug: helpers with conditional returns are known, 91 of the corpus) | T5 |

## 6. Expectations, stated before any run

| measurement | expected |
|---|---|
| of 24 Opus kernels: parse; validate and compile on Metal | 20–24; 16–22 |
| read into the safe kingdom (the largest risk: heaps, hash emulation, nested helpers) | 2–10 before T5 |
| valid on every row; mean recall ≥ 0.95 | 4–12; 2–8 |
| faster than the CPU's wall, generation 0 | 0–3 |
| after 20 generations: best speed-up over the CPU at equal recall | 2–10x |
| the CPU's propose wall for one 1,000-row window (all cores) | 0.1–1 s; measured by M1 before any budget is set |
| wall clock: T1–T4; T5; T6 | 4 h; 2–6 h; 3 x 45 min, the budget set from M1's wall (the ABBA timing rounds multiply each case) |

## 7. Tasks

1. M1 capture (after approval), its output checked against `ingest_window` by re-running
   the window from the dumped state on the CPU and comparing lists bit for bit.
2. M2 and M3 with tests (the CPU's own lists pass at recall 1.0; a list with a wrong id
   fails validity; a shuffled list passes).
3. M4.
4. The 24 seeds, no evolution: the ladder counts against section 6.
5. M5 until at least 8 seeds enter the kingdom.
6. Evolution, three seeds; report against section 6.
7. The end-to-end gate.
8. Problem 2 (seat, commit, advance), the same way.

## 8. Risks

- The safe reader accepts 1,653 of 1,798 Sonnet kernels written for Gram, nluca and
  k-means; a settle kernel has data-dependent loops, private heaps and 64-bit hash
  arithmetic in u32 pairs. Each refusal is our bug to fix (M5), and T5 may be long.
- A thread-per-row walk keeps heaps and seen-sets of hundreds of entries: private memory
  pressure on Metal; workgroup-per-row designs avoid it. The 24 strategies cover both.
- The fitness rewards agreement with one algorithm's output, and a different graph
  builder can serve as well: the end-to-end gate measures serving, the route and list
  agreement measure faithfulness; both are reported, and a faster kernel that serves
  as well but agrees less is reported as that, not as the CPU settle.
- Problem 1 is judged against a frozen state, while the stream couples windows (seat,
  commit and advance build the next window's state; the rolling correction re-proposes
  against the matured state and flushes its tail). T7 runs the GPU proposals inside
  the CPU stream, correction included, so the coupling is measured before problem 2.
- A long or looping kernel cannot be cancelled once dispatched: the budget's
  `gpu_wait_ms` and the static cost bound apply, and a killed run is followed by a
  shipped-kernel probe.
