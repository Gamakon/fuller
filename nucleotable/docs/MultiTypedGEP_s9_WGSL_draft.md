# 9. WGSL legality as types (draft for revision 5)

Draft of §9 of `MultiTypedGEP.md`, task D1 of `docs/PLAN_true_gep_wgsl.md` (revision 4), written 2026-10-21 against `docs/GOAL.md`, rev 4 §1–§7, the rule ledger, the five reviews, the A1 and A3 reports, `kingdoms/safewgsl/gen_tables.py` (post-A0 working tree), `src/wgsl/{check,safety,render,harness,safe_reader}.rs`, `src/typed_gep.rs` and `src/kingdom_contract.rs`. It is the specification the build (B1, B2, B3, C1, C2) is held to. Open questions are marked **OPEN** and collected in §9.7.

**What this section is.** The WGSL kingdom is a family of kingdoms, one per contract. A kingdom `K` of the family is an ordinary kingdom of §1: a symbol table whose rows have one output type and ordered input types, checked once, with the layouts, bounds, walk and operators of §2–§6 unchanged. Everything WGSL-specific is in *which* types and rows `K` has. The legality rules of `check.rs`, `safety.rs` and the harness are each a statement about the rows of `K`, true of every tree the walk of §3 can produce, and the renderer spells the tree. The sections below give, for every rule, the types, the rows that produce and consume them, the closure argument, the expressiveness cost, and the rows of today's `kingdoms/safewgsl` that go.

**The one closure lemma, used by every rule.** Under the walk of §3 every node of an expressed tree is a row of `K` whose output type is the type its parent's slot demands (frame membership), and under §5 every child of every operator is again a gene of the same layout over the same alphabet. So for any type `T`, every `T`-valued subtree of every expressed tree is rooted at a row of `K` whose output is `T`. A rule is therefore proved for all expressed trees by exhibiting the complete list of rows outputting each type it mentions and checking the rule on those rows and their declared spellings. No argument below needs anything about the gene's bytes.

## 9.1 The contract-instantiated kingdom `K<target, interface, dispatch contract, alias partition>`

### 9.1.1 The contract

A contract `C` is one text file per `(problem, entry point)`, the format of `safety::Contract::parse` (`problems/*/safety.contract`) widened by the lines below. Its facts are everything the rows of `K` are generated from. Nothing in a gene restates a contract fact; the renderer spells them once from `C`.

| line | fact | today's source |
|---|---|---|
| `target <name>` | the device: naga capabilities, `max_invocations`, `max_size`, `max_workgroup_storage` (`Target`) | `Target::recorded` / `of_device` |
| `interface <name>` | the binding schema in `@group(0)` order: `(binding, name, kind, element or fields)`; the uniform variable and its fields; the entry point's name | `ProblemSpec.binding`, `KernelSpec` |
| `entry <name>` | the entry point this `K` is for; a module of several entry points is a chromosome of several genes (§9.1.4) | `KernelSpec.entry`, `entries` |
| `workgroups x|y|z lo hi` | the dispatch geometry of this entry | as today |
| `param <name> lo hi` | the admissible range of every uniform field and every `arrayLength(b)` | as today |
| `length <b> <poly>`, `full-length` | a lower bound on a binding's length, a polynomial in the params | as today |
| `alias <b> <buffer> <offset>` | the alias partition: bindings on one buffer, with offsets | as today |
| `writes <b>...` | the bindings this entry stores to (its outputs) | derived today from the role's `store.<b>_*` rows |
| `primary <b>` | the output the epilogue ends in (§9.2.6); absent: no epilogue | the Gram role's `h` |
| `partition contiguous|strided|grouped <G>` | how a thread's block of an output is laid out (§9.2.3) | the nine `cell_of` pairs |
| `mode <m>...` | the modes the problem lists; `K` is instantiated per listed mode (`K<..., mode>`) | `KernelSpec.modes`, A1 H1 |
| `passes lo hi` | the `// passes N` range when the problem reads it | `KernelSpec.passes_header` |
| `policy strict|arena <b>...|rows <b> <w>` | layer 7's policy, named: which bindings are arenas, which are row-owned | `--layer7`, `Contract.arena`, `rows` |
| `determinism strict|relaxed` | whether an atomic's returned old value may be consumed (§9.2.12) | none (Codex rev4 finding 2) |
| `wgmap <name>` | the workgroup-memory map: a fixed list of named arrays with element type and length, total within the target (§9.2.2) | `wg.decls` knobs, the renderer's halving |
| `budget <K>` | the root fuel stratum `K` in `0..=4` (§9.2.8): work per invocation at most `N · 2^(6K)` node executions | none (the deadline) |

### 9.1.2 Validation on the host, before any gene exists

Instantiation runs `validate(C)` once. It is input validation of a human-written file; it never sees a gene. Every defect is "our software has a bug: ..." and stops the instantiation. It establishes:

1. every `param` range is non-empty and within `u32`; every uniform field of the interface has a `param` line; every `arrayLength(b)` of a written binding has a `length` polynomial that is positive over the ranges;
2. `workgroups` per axis within the target's dispatch limits; the target's limits positive;
3. the alias partition is a partition: two bindings in one class are one buffer; a class with a written member has no two members whose ranges overlap unless the contract declares them identical (same offset and length); every binding not named is its own class;
4. `writes` names read_write bindings only; `primary` is one of `writes` or absent; arena and rows bindings are read_write; a binding is in at most one of arena, rows, strict;
5. every `mode` listed is one the harness lists for the problem; `passes` within `1..=16`;
6. the `wgmap`'s arrays, each rounded up to 16 bytes as Metal allocates them, total at most `max_workgroup_storage`;
7. `budget K ≤ 4`, and the wall bound of §9.2.8 at this `K`, this dispatch and this target is at most the problem's `base_ms`;
8. the feature probes (§9.2.1): for each feature row family the target claims (f16, subgroups, subgroup barrier, i32 division, shifts), a probe kernel of that family is compiled and dispatched on the target once; a family whose probe fails is absent from `K` and the instantiation report says so.

### 9.1.3 Generating the rows

`gen_tables.py` becomes a function of the contract: `gen_tables.py --contract <file>` writes `kingdoms/safewgsl/<target>/<interface>/<entry>/{types.tsv, functions.tsv, TYPES.md, symbols.md}`. The generated rows are the fixed families of §9.2 plus the contract-derived families: the own-cell rows of each written binding (§9.2.3), the arena index rows (§9.2.4), the workgroup map's array terminals and index duals (§9.2.2), the shape terminals of the target (§9.2.1), the knob references, the literal domains that depend on the contract (`u32.boundm` and friends carry the param's range). Then the kingdom check of §1 runs (`safewgsl_relations` on the generated tables), then the bound table of §2 for the layout's `N`, and the generator's own tests of §9.6.

A generated kingdom is immutable for an epoch. Changing the contract makes a new kingdom; a population's genes are re-encoded through the write-back of §5 or discarded, never patched.

### 9.1.4 Modules are chromosomes

The interface's entry points (`ingest`, `search`) have different dispatch geometries and therefore different ownership rows. Each entry point is its own kingdom `K_e`, and a gene of `K_e` is rooted at `kernel.safe` of `K_e`. A problem with several entry points is a chromosome whose normal genes are one gene of each `K_e` and whose linker is a **fixed** homeotic gene (§6): `head_R = (kernel.module_of_<interface>)`, tails the genic terminals in the interface's order, `R = kernel.module<interface>`. `genes(chromosome_id, gene_id, kingdom_id, layout_id)` already carries the kingdom per gene. The module row's spelling is the shared prologue (the contract's bindings, uniform struct, helpers, `enable` directives) followed by each entry's text in order. The rows `kernel.module_2/3/4`, `kernel.module_ext1..3`, `own.kernel_module` and the dual `kernel.module` leave the per-entry tables. This resolves `module_entries_not_the_interfaces` and `root_not_safe_kernel` (§9.2.13).

## 9.2 Every rule of the ledger as types

Each subsection: **types** (duals in `slot.form` style; a parameter in the form is spelled into the instance, e.g. `u32.own_cands`, `stmt.L2_f1`), **rows** (producers and consumers), **closure**, **cost**, **removed**. The ledger row of the plan each subsection settles is named in its heading; the `check.rs` and `safety.rs` reasons it covers are listed, and §9.8 indexes every reason back to its subsection.

### 9.2.1 Contract, target limits, features (layer 5; harness H1, H2)

Reasons: `workgroup_size_over_limit`, `workgroup_memory_over_limit`, `feature_missing`; H1 (`// mode`, block factors `1..=16`, `// grid` header equal to the workgroup, every binding in the schema); H2 (pipeline creation); R3 (workgroup over 1,024, arrays over 32 KB, f16 without `enable f16`).

**Types.** `grid.shape` — a constant-capable dual whose literal domain is the table of the target's legal workgroup shapes `(x, y, 1)`: every pair with `1 ≤ x ≤ max_size[0]`, `1 ≤ y ≤ max_size[1]`, `x · y ≤ max_invocations` (7,690 shapes on the M3 Max). A `Dc_shape` word is an index into that table; `?_shape` is its placeholder. `u32.blk` keeps its domain `{1, 2, 4, 8}` (within H1's `1..=16`). `u32.passes` with domain `passes lo hi`, present only when the contract has the line. `grid.grid` keeps its dual.

**Rows.** `grid.of: (grid.shape, u32.blk, u32.blk) -> grid.grid`, `grid.of_rows_x` likewise; `grid.with_ktile: (grid.grid, u32.lit4096) -> grid.grid`; `grid.with_passes: (grid.grid, u32.passes) -> grid.grid` (only with a `passes` line). `own.grid_grid` is the shape `(16, 16, 1)` at block `1 x 1` where that shape is legal on the target, else the table's first entry. Feature families are present only where the target's probe passed: the `f16.*` and `vec*<f16>.*` duals and every row over them; the `subgroup.*` rows and `builtin.subgroup_*`; `stmt.subgroup_barrier_*`. The module text begins `enable f16;` whenever `K` has the f16 duals.

**Closure.** The only producer of `grid.grid` from values is `grid.of`/`grid.of_rows_x` over `grid.shape`, whose every value is a legal shape of the target by the table's construction, so `@workgroup_size` is within the limits for every tree. Workgroup memory is the map of §9.2.2, fixed within the limit at validation. A feature row exists in `K` only if the target compiled and ran its probe, so `feature_missing` and H2 have no tree to fire on; a creation failure in C1's dispatch stage is a probe that lied, a kingdom bug of the target. The `// grid x y bx by` line is spelled from the grid node and the `@workgroup_size` from the same node, so they agree; `// mode m` and `// passes N` are spelled from the contract and the grid node, within the listed modes and the passes range; the prologue is the schema, and no row declares a binding.

**Cost.** One kingdom per `(target, mode)`; non-power-of-two shapes are in the table, so nothing is lost there. **OPEN 1:** the shape table as a 7,690-entry literal domain versus a smaller powers-of-two table (66 entries); the seeds (S1) decide.

**Removed.** `literal.wgdim`, `u32.wgdim`, `workgroup_dim`, `workgroup_shape`, `snap` on block factors (renderer S1, S2), `Target::check_limits` on expressed kernels (oracle only), `features()` of layer 5 (oracle only), the mirror rows from every `K` whose mode is not `full` (§9.2.3).

### 9.2.2 Array range: workgroup, private, storage reads (layer 3, `index_out_of_range`; the `wg.decls` grammar)

Reasons: `index_out_of_range`, `workgroup_array_not_in_grammar`, duplicate `decls`, `index_not_in_bounds` for reads; renderer S7 (array re-masking), R4 (a workgroup array used as a value).

**Types.** The workgroup-memory map `wgmap` of the contract fixes a list of arrays, each `(name, element, length 2^j)`. Every array is a terminal of a dual that carries its element and length class: `wgmem<f32,2048>.real`, `wgmem<vec4<f32>,512>.real`, `wgmem<u32,1024>.index`, `wgmem<i32,1024>.int`, `wgmem<atomic<u32>,256>.index`, `wgmem<atomic<i32>,256>.int`. One index dual per length class: `u32.wgidx_<j>` ("an index below `2^j`, by the mask its row writes"), and `u32.wgown_<j>` ("the element `lidx + k · threads` this thread owns, below `2^j`"). Private arrays likewise: `parr<f32,64>.real`, `parr<vec4<f32>,16>.real`, `parr<u32,16>.index`, `parr<u32,64>.index`, `parr<u32,128>.index`, `parr<bool,16>.flag`, `parr<vec2<f32>,16>.real`, with `u32.pidx_<j>`.

**Rows.** `safe.wg_of_<j>: u32.index -> u32.wgidx_<j>` spelled `(i & (2^j - 1)u)`; `safe.pidx_<j>: u32.index -> u32.pidx_<j>` spelled the same; `safe.wg_own_<j>: u32.lit64 -> u32.wgown_<j>` spelled `min(lidx + k * THREADS, (2^j - 1)u)` and `safe.wg_own_tile_<j>: u32.tcounter<d> -> u32.wgown_<j>` likewise (`THREADS` is the grid knob). Every read of an array of class `j` takes `u32.wgidx_<j>` or `u32.wgown_<j>`: `wg.get_<e>_<j>: (wgmem<e,2^j>, u32.wgidx_<j>) -> e`; every write takes `u32.wgown_<j>` (§9.2.5). Private arrays: `parr.get_<e>_<j>: (parr<e,2^j>, u32.pidx_<j>) -> e`, `parr.set_<e>_<j>_<rung>: (parr<e,2^j>, u32.pidx_<j>, e) -> stmt.<rung>`. Lane accesses (`get_v_lane`, `set_v_lane`) spell `[(j) & 3u]` as their declared text. Declarations are spelled from the map, once, in the prologue; a gene has no declaration rows.

Storage reads: for every storage binding `b` of the interface, `load.<b>: u32.index -> <elem>` spelled `b[i]`, whose declared meaning is WGSL's (a read past the end returns the last element under naga's `Restrict` policy, the one the harness's device installs). Reads of an output binding are at owned cells only (§9.2.3); reads of an arena at arena indices (§9.2.4).

**Closure.** An array of class `j` is accessed only by rows whose index input is `u32.wgidx_<j>` / `u32.wgown_<j>` / `u32.pidx_<j>`, and those duals are produced only by the mask and own rows above, whose text is below `2^j` for any operand. The array's length is `2^j` by the map, so the index is in range for the array it indexes (GOAL item 4, first bullet). Two arrays never share a dual with different lengths, which was A3 3.2's hole. The map's total is within the target by validation, so no plan is ever built or halved.

**Cost.** A kernel's workgroup arrays are chosen from the map, and lengths are powers of two; a problem whose seeds need another map declares it (S1 derives the maps from the seeds). A read past a read-only binding's end returns its last element: that is the row's meaning, not an in-range index. **OPEN 2:** the fixed map versus a typed buddy-tree of declarations (`wg.decls_<k>` strata that pack any power-of-two sizes within `2^k` bytes), which keeps lengths evolvable at the cost of about 60 more duals; the map is chosen here because declared-length knobs were the renderer's S2/S7 repairs. **OPEN 3:** whether Andrew accepts "last element" as the meaning of a read past a read-only binding, or wants input reads at a proved index too (a `u32.idx_<b>` family built like §9.2.3 from the length polynomial, which costs the free `phi[t*d + c]` addressing).

**Removed.** `u32.wgindex/wgvindex/wgv2index/wgaindex/pindex/pindex16/pindex128` (replaced by the class duals), `safe.wg_of/wgv_of/wgv2_of/wga_of/pidx64/pidx16/pidx128`, `u32.wglen`, `literal.wglen`, `wg.decl*`, `wg.decls*`, `grid.with_wg`, `wg.decl` and `wg.decls` duals, the raw `wg.get`/`wg.set`, `load.buffer` (F007), `u32.phi`, `load.phi_safe` (now `load.<input>`), `workgroup_plan`, `array_len` lookups in the renderer, the renderer's re-mask (S7), `own.u32_wgindex` and friends (now `own.u32_wgidx_<j>` = `0u`).

### 9.2.3 Write ownership and disjointness per output binding (layer 7; layer 3 `store_cell_wraps`)

Reasons: `write_sets_overlap`, `store_index_not_affine`, `store_index_may_wrap`, `store_stride_may_be_zero`, `index_not_in_bounds` (writes), `store_cell_wraps`, `output_read_of_another_invocations_cell`; the mirror (`u32.mirror`, `diag_twice`); A3 2.6's mixed block-form pairs; renderer S4 (fallback own leaves), S8 (offset clamps).

**Types.** For each written binding `b` of the entry that is neither arena nor rows (the alias class stands for its members): `u32.own_<b>` — "a cell of `b` this invocation owns". The thread's origins: `u32.rowbase`, `u32.colbase` (the first row and column of the thread's block, under the contract's partition), and `u32.threadrow`, `u32.threadcol` (the origin plus a clamped offset). The six partition-specific origin duals collapse into these two, because the partition is a contract fact spelled by the rows, not a choice made per node.

**Rows.** Origins: `safe.thread_row: vec3<u32>.gid -> u32.rowbase` and `safe.thread_col: vec3<u32>.gid -> u32.colbase`, spelled by the partition: contiguous `gid.y * BROW`, strided `wid.y * (WROW * BROW) + lid.y`, grouped `wid.y * (WROW * BROW) + lid.y * (BROW / G)` (the knobs `BROW`, `WROW`, `G` are the grid's and the contract's). Offsets, each taking an origin and clamped into the block, as today: `safe.row_plus: (u32.rowbase, u32.lit64) -> u32.threadrow` spelled `o + min(i, BROW - 1u)` (contiguous), `o + min(i, BROW - 1u) * WROW` (strided), `o + min(g, G - 1u) * (WROW * BROW / G) + min(e, BROW / G - 1u)` (grouped, `safe.row_plus_g` with two offsets); the tile-counter and any-expression offsets `safe.row_plus_tile<d>`, `safe.row_plus_x` likewise; `safe.row_as_index: u32.threadrow -> u32.index` and `safe.rowbase_as_row: u32.rowbase -> u32.threadrow` (the origin as a row, offset zero). Cells: for a 2-D output `b` with the contract's row stride `S_b` (a param, `cfg.d`), `safe.cell_<b>: (u32.threadrow, u32.threadcol) -> u32.own_<b>` spelled `r * S_b + c`; for a 1-D output `safe.cell_<b>: u32.threadcol -> u32.own_<b>` spelled `c`. Stores: `store.<b>_<rung>: (u32.own_<b>, <elem>) -> stmt.<rung>` spelled `if (G_b) { b[i] = v; }` where `G_b` is the guard the generator proves sufficient (§9.6.2): the identity (no guard) when the template is in bounds for every value of the params, else `r < R_b && c < C_b` from the length polynomial (the Gram's `row < r_rows && col < d`). Accumulating stores `store.<b>_add_<rung>` spell `b[i] = b[i] + v`. Loads of an output: `load.<b>_own: u32.own_<b> -> <elem>` spelled `b[i]`, the only load row of a written binding. Own terminals: `own.u32_rowbase`, `own.u32_colbase` (the origins themselves), `own.u32_threadrow` (origin, offset 0), `own.u32_own_<b>` (the thread's first cell).

The mirror: in `K<..., mode=full>` only, `u32.mirror_<b>` with `safe.mirror_<b>: (u32.threadrow, u32.threadcol) -> u32.mirror_<b>` spelled `c * S_b + r`, and `store.<b>_mirror_<rung>: (u32.mirror_<b>, <elem>)` spelled `if (r < c && G_b) { b[c * S_b + r] = v; }`: the triangle guard is the row's text, so the diagonal is written by the own store only.

**Closure.** `u32.own_<b>` is produced only by `safe.cell_<b>` over `u32.threadrow` × `u32.threadcol`, which are produced only by the origin and offset rows, which read the thread's coordinates and the grid's knobs. So every store index of `b` in every tree is `origin(thread) + clamped offset`, with one partition and one block factor for the whole kernel (both are grid facts spelled into every origin row, so two store sites of one binding in one kernel share them). The generator proves, per contract, once, that the set of cells `{origin(t) + o : 0 ≤ o < block}` is disjoint across invocations `t` of the dispatch and that `r * S_b + c` does not wrap over the param ranges, by interval arithmetic over the contract and by the exhaustive enumeration of §9.6.2 at the contract's smallest shapes; a template that fails is not emitted and the instantiation report names it. Hence `write_sets_overlap`, `store_index_not_affine`, `store_stride_may_be_zero`, `store_index_may_wrap` and `store_cell_wraps` have no tree to fire on, and `index_not_in_bounds` is met by `G_b`. The only load of `b` takes `u32.own_<b>`, so no invocation reads a cell another writes. The mixed-pair hole of A3 2.6 closes because the partition is one per kernel.

**Cost.** Stores to a non-arena output are at `thread origin + bounded offset` under one partition per kernel (contiguous, strided or grouped); scatter and data-dependent stores go to arenas (§9.2.4) or through atomics; a transposed write is the mirror template under `mode full` only; two partitions in one kernel are unspellable. Exactly the price layer 7's Revision 1 already charges (non-affine stores illegal). **OPEN 4:** the per-entry `writes` list of the V24 contracts (whether `ingest` writes `cands`) must come from the problem; the instantiation refuses an entry whose written binding has no partition fact.

**Removed.** `u32.cell`, the nine `safe.cell_of` pairs, `u32.threadrows/threadcols/threadrowg/threadcolg`, `u32.rowbases/colbases/rowbaseg/colbaseg`, `safe.thread_row_s/_g`, `safe.*_s` and `_g` offset rows (one family under the partition fact), `store.h_*`, `load.h_safe`, `load.h`, `store.h` (role), `store.buffer` (F195), `u32.blk` as an argument of the origin rows (it is the grid's), `u32.mirror` and its rows outside `mode full`, `render::own_fallback`, `fallback_value`, `own_leaf` as a substitute (S4), `Contract::for_role`'s layer 7 on the evolution path, the renderer's offset clamps as a separate step (S8: the clamp is the row's text).

### 9.2.4 Arenas and row-owned outputs: the policy as a contract choice

Reasons: `arena_index_not_in_bounds`, the `--layer7 strict | arena | arena+rows` policy (A1 P6), `index_not_in_bounds` on arena bindings, `rows` mode.

**Types.** For each arena binding `a` (`policy arena a`): `u32.aidx_<a>` — "an index below `arrayLength(a)`, by the mask or modulus its row writes"; the contract's `param arrayLength(a) L L` fixes `L`. For each rows binding `r` (`policy rows r W`): `u32.ownrow_<r>` — "a cell of the row this workgroup owns".

**Rows.** `arena.idx_<a>: u32.index -> u32.aidx_<a>` spelled `(i & (L - 1)u)` when `L` is a power of two (V24: `8,388,608 = 2^23`), else `(i % Lu)`; `load.<a>: u32.aidx_<a> -> <elem>` (`a[i]`, or `atomicLoad(&a[i])` for an atomic array); `store.<a>_<rung>: (u32.aidx_<a>, <elem>) -> stmt.<rung>`; `atomic.<a>_<fun>: (u32.aidx_<a>, <elem>) -> <atomic result>` (§9.2.12). Rows binding: `safe.ownrow_<r>: u32.index -> u32.ownrow_<r>` spelled `R * W + min(e, W - 1)` where `R` is the workgroup's row (`wid.x`, the contract's `workgroups x 1 n_queries`); `store.<r>_<rung>: (u32.ownrow_<r>, <elem>)`, `load.<r>_own: u32.ownrow_<r> -> <elem>`.

**Closure.** Every arena access is at an index below `L` by its row's text; every rows-binding store is inside the workgroup's own row of `W` cells. Two workgroups never write one cell of `r`; invocations of one workgroup may, and the contract declares that the design orders them (atomics, phases) and that fitness measures the result, exactly as today's rows mode does. Plain stores to an arena at colliding indices give WGSL's indeterminate value, not a fault; the contract declares arenas as the design's own state.

**Cost.** The arena and rows policies are kingdoms, not flags: a kernel of `K<policy strict>` cannot touch an arena. A rows binding's cells within a row are not proved disjoint across the workgroup's invocations (declared).

**Removed.** `Contract.arena_mode`, `rows_mode`, `--layer7`, `kernel_info`'s mode handling of these; the role's `store.<name>_<rung>` and `load.<name>` at `u32.index` for atomic arrays (now at `u32.aidx_<a>`).

### 9.2.5 Workgroup memory races and ordering: phases (layer 7)

Reasons: `workgroup_memory_race`, `memory_barrier_without_control`; renderer S10 (statement erasure), `wg.put`'s double barrier.

**Types.** Two new block kinds at every depth `d`: `block.W<d>` / `stmt.W<d>` — "a write phase: statements between a leading and a middle barrier; writes to workgroup arrays at this thread's own elements only, reads of workgroup arrays at this thread's own elements only, no barrier"; `block.WB<d>` — a write-phase loop body that breaks. The existing `L`, `C`, `O`, `B` kinds keep their meanings and gain: "reads workgroup arrays at any index, writes none".

**Rows.** The phase rows, at the barrier rungs only: `wg.phase_L<d>: (block.W<d>, block.L<d>) -> stmt.L<d>` spelled `workgroupBarrier(); { W } workgroupBarrier(); { L }`, and `wg.phase_s_L<d>` whose separators are `storageBarrier(); workgroupBarrier();` (the storage barrier is thus always inside a control barrier row). `wg.put_<e>_<j>_L<d>: (wgmem<e,2^j>, u32.wgown_<j>, e) -> stmt.L<d>` keeps its spelling `workgroupBarrier(); if (i < len) { a[i] = v; } workgroupBarrier();` (a one-write phase). Writes: `wg.set_<e>_<j>_W<d>: (wgmem<e,2^j>, u32.wgown_<j>, e) -> stmt.W<d>` and the lane form; the only writers of workgroup arrays, and they output `stmt.W<d>` only. Own reads in W: `wg.get_own_<e>_<j>: (wgmem<e,2^j>, u32.wgown_<j>) -> e`, the only workgroup read row W-statements can hold (the generator gives `block.W<d>` statement rows a value alphabet in which `wg.get_<e>_<j>` over `u32.wgidx_<j>` is absent: W-rung register and private-array statements take values of the W value family; see the note below). Foreign reads: `wg.get_<e>_<j>` over `u32.wgidx_<j>` anywhere outside W. Reductions `wg.reduce_*_L<d>`, `wg.uload_*_L<d>` keep their spellings (each begins with a barrier). Atomic workgroup rows are usable in every kind. Loops in W bodies take `block.W<d+1>` / `block.WB<d+1>` bodies; `if` in W takes `block.W<d>` bodies.

Note on value typing inside W: values are rung-free in this design (a `f32.real` is one dual everywhere). To keep foreign workgroup reads out of W, the W kind's statement rows take their values from a second value family: `f32.realw`, `u32.indexw`, ... — **OPEN 5**: this doubles the value duals (about 160 more types) and the width; the alternative is a W kind whose statements are a restricted list (register sets from own reads and arithmetic over registers, private-array sets, own writes) with the value family shared but the foreign-read rows typed `f32.realr` ... each choice costs one extra value family. The build takes the cheaper measured one in B1; this draft specifies the "W values are a family" form and counts it in §9.3.

**Closure.** The spelling `barrier; W; barrier; L` orders every write (in W) after every read of the preceding L code and before every read of the following L code. Within W, writes are at `u32.wgown_<j>` (distinct per invocation by `lidx + k · threads` below the length; `k` a literal or a uniform tile counter, so two invocations of one workgroup never write one element), and the only reads are own reads, so no element is both read by one invocation and written by another between two barriers. The reductions, puts and uniform loads carry their own barriers in uniform control (L rungs). Barriers exist only in rows outputting `stmt.L<d>`, so a memory barrier always sits in a control barrier row. Hence no tree has a workgroup race.

**Cost.** A read-modify-write of a foreign element within one phase is unspellable; an unsynchronised shared pattern (one invocation writes, another reads without a barrier) is unspellable, as intended. Two more block kinds per depth.

**Removed.** `wg.set_<r>` over `u32.wgindex` (any masked index) at every rung, `wg.set_v_lane_<r>` likewise, `stmt.storage_barrier_<r>` as a free statement, `stmt.barrier_<r>` as a free statement (a barrier is a phase separator; **OPEN 6:** keep `stmt.barrier_L<d>` as a degenerate phase with an empty W, spelled `workgroupBarrier();`, for seeds that barrier without writing), `render::scope.masked` and the erasure branches (S10), `Rendered.masked`, `is_barrier` in the oracle's grammar layer.

### 9.2.6 Statement grammar, rungs, barriers and uniformity (layers 2, 4; layer 7 uniformity rules)

Reasons: `barrier_under_condition`, `break_outside_loop`, `continue_outside_loop`, `block_not_in_safe_grammar`, `statement_not_in_safe_grammar`, `loop_not_bounded_constructor`, `nested_item_loop`, `tile_depth_exceeded`, `barrier_in_nonuniform_control_flow`, `barrier_in_nonuniform_control`, `subgroup_operation_in_nonuniform_control`, `loop_bound_not_uniform`; renderer R2.

**Types.** As today: `stmt.L<d>`, `block.L<d>` (`d` in `0..=5`), `stmt.C<d>`, `block.C<d>`, `stmt.O<d>`, `block.O<d>` (`d` in `0..=4`), `block.B<d>` (`1..=5`), plus `W<d>`, `WB<d>` of §9.2.5, each at the fuel strata of §9.2.8 (`_f<k>`). `u32.uniform`, `bool.uflag`, `u32.boundu` as today. Loop counters split by uniformity: `u32.counter` (the item loop with a uniform bound) and `u32.xcounter0` (the item loop with an expression bound); `u32.tcounter<d>` (a tile loop at an L or O rung whose bound is a literal or `u32.boundu`: uniform) and `u32.xcounter<d>` (a tile loop at a C, B or W rung, or any loop that breaks: divergent).

**Rows.** As today for blocks, `if`, `if_else`, `cblock`, `return_L0..2`, `cont_*`, `brk_B<d>`, loops at every rung, the `if_uflag`, `if_else_uflag`, `if_u{le,eq,ne,lt}` uniform tests, `uflag.*`, `uniform.*` (post-A0: no `of_flag`, no `of_checked`); loop rows at L and O rungs take `u32.lit64`/`u32.lit4096`/`u32.boundu`/`u32.boundm` and bind `u32.tcounter<d>`; loop rows at C, B, W rungs and every `brk` loop take any bound (`u32.boundx<c>`, §9.2.8) and bind `u32.xcounter<d>`; `uniform.of_counter: u32.counter -> u32.uniform`, `uniform.of_tcounter<d>: u32.tcounter<d> -> u32.uniform` (no row takes an `xcounter` into `uniform`). Counter reads: `safe.tile_counter<d>: -> u32.tcounter<d>` and `safe.xtile_counter<d>: -> u32.xcounter<d>`, each spelled as its variable, declared at kernel scope and initialised `0u`: before its loop it reads `0u`, inside it the trip, after it the final value (**OPEN 7:** today's renderer writes `lit_u32(0u)` for a read outside the loop, S5; the variable read is the pure spelling, and the reader reads a kernel's counter use after its loop as the final value, exactly).

Subgroup operations become statement rows at the L rungs: `subgroup.<op>_<reg>_L<d>: (reg<T>, T) -> stmt.L<d>` spelled `r = subgroupOp(v);` for every collective and gather (`add`, `mul`, `min`, `max`, inclusive and exclusive scans, `and`, `or`, `xor`, `all`, `any`, `broadcast`, `broadcast_first`, `shuffle*`, `quad_*`, `ballot`), one per register kind the result has; `stmt.subgroup_barrier_L<d>` as today. The subgroup builtins (`subgroup_invocation_id`, `subgroup_size`, `num_subgroups`, `subgroup_id`) stay value terminals.

**Closure.** The rung structure is unchanged from A3's YES rows: a `C`, `B`, `W` block holds no row that outputs a barrier-bearing statement; every `if` over a `bool.flag` drops to a `C` body; every `if` that keeps the rung takes `bool.uflag` or two `u32.uniform`, whose producers are the contract's uniform fields, the workgroup id, the grid, literals, arithmetic of uniforms, and uniform counters; loops that keep an L body take uniform bounds; loops whose body is `C`/`B`/`W` bind `xcounter`, which no row makes uniform. A barrier (phase, put, reduce, uload, subgroup barrier, subgroup statement) is a row outputting `stmt.L<d>` only, so it is reached only through L blocks, whose every enclosing condition and bound is uniform. A conditional return has no row. Nesting (`nested_item_loop`, `tile_depth_exceeded`) is the rung ladder: no row takes `block.L6`. Subgroup operations are statements at L rungs, so they are in uniform control.

**Cost.** A subgroup result passes through a register (one statement) instead of sitting in an expression; a counter of a divergent loop is never a uniform value (it was never one).

**Removed.** `uflag.of_flag`, `uniform.of_checked` (A0), `subgroup.*` value rows (71 templates), `uniform.of_tcounter<d>` over a C-rung loop's counter (split), the oracle's grammar layer on the path, R2's renderer branches.

### 9.2.7 Bounded loops and the body-invariant bound (layer 6)

Reasons: `loop_bound_wraps`, `loop_bound_changes_in_body`; renderer S6 (loop caps).

**Types.** Trip classes: `u32.lit64` (`1..=64`), `u32.lit4096` (`1..=4096`), `u32.boundm`/`u32.boundd`/`u32.boundrows` (each present only when the contract has the param, carrying the param's range), `u32.boundu` (uniform, capped), `u32.boundx6`, `u32.boundx12`, `u32.boundx18`, `u32.boundx24` (an expression bound capped at `2^c`), `u32.xbound` as today (the value of an expression bound).

**Rows.** `bound.of_index_<c>: u32.xbound -> u32.boundx<c>` spelled `min(e, 2^c u)`; `bound.of_uniform_<c>: u32.uniform -> u32.boundu<c>` likewise; `bound.m: -> u32.boundm` spelled `cfg.m`. Every loop row spells its bound into a `let` before the loop: `let n<d> = <bound>; for (var k<d> = 0u; k<d> < n<d>; k<d> = k<d> + s) { body }`, so the trip count is fixed at entry whatever the body writes. Strides `s` are literals in `1..=4096`.

**Closure.** The bound is read once, before the body, into an immutable `let`, so `loop_bound_changes_in_body` has no tree to fire on. The counter steps by `s ≤ 4096` to a bound of at most `2^24` (expression and uniform bounds) or at most the contract's param range (a binding bound, whose range the generator checks is at most `2^32 - 4096`, else the bound row is capped like an expression bound), so the counter never steps past `u32::MAX`: `loop_bound_wraps` has no tree. The `xbound` rows cannot grow past their operands except `index.add`/`max`/`or` of two bounds, which may wrap to a *smaller* value: a smaller bound is still a bound, and the cap row makes it finite.

**Cost.** Trip counts above `2^24` are unspellable; a loop's bound is evaluated once (a loop whose bound a seed changes inside its body reads as outside the kingdom by reason, C4).

**Removed.** `MAX_LOOP_TRIP` as a renderer constant (the cap is the row's text), the single `u32.boundx`, `bound.of_index`, `bound.x_of_uniform` (uncapped) and `wg.get_u_safe -> u32.xbound` (a workgroup read as a bound value stays, through `u32.xbound`), the oracle's layer 6 on the path.

### 9.2.8 The execution budget: fuel strata (the Metal hang, the deadline, predicted time, the runaway drain)

Reasons: A1 P3 (exit code 3), H6 (`slow: predicted`), H10 (`runaway`), the cost estimator (`render::cost`), the harness deadline.

**Types.** Every statement-level dual carries a fuel stratum `k` in `0..=4`: `stmt.L2_f1`, `block.C3_f2`, `block.W1_f0`, ... Stratum `k` means: executing this statement once performs at most `N · 2^(6k)` node executions, where `N` is the layout's program size (every expressed function executes at most once per iteration of its enclosing loops, and a loop-free subtree has at most `N` functions). Trip classes: `c1 = 2^6` (`lit64`, `boundx6`, `boundu6`), `c2 = 2^12` (`lit4096`, `boundx12`, `boundu12`, and a binding bound whose contract range is at most `2^12`), `c3 = 2^18` (`boundx18`, `boundu18`, binding bounds up to `2^18`), `c4 = 2^24` (`boundx24`, `boundu24`, binding bounds up to `2^24`; a binding bound above `2^24` has no row).

**Rows.** A loop row of trip class `c_i` over a body of stratum `k` outputs stratum `k + i`, and exists only when `k + i ≤ 4`: `loop.tile_L0_f2: (u32.lit64, block.L2_f1) -> stmt.L0_f2`, `loop.items_boundm_f3: (u32.boundm, block.L1_f1) -> stmt.L0_f3` (with `cfg.m ≤ 2^12`), and so on for every loop row of §9.2.6 including unrolled loops (an unrolled body is `n` copies: the same class). Sequence rows take statements of one stratum: `block.seq4_L0_f2: (stmt.L0_f2 × 4) -> block.L0_f2`; a lift row raises a statement one stratum without changing its text: `stmt.lift_L0_f2: stmt.L0_f1 -> stmt.L0_f2`. `if`, `if_else`, `cblock`, `block_`, `cont_`, `brk_`, the phase rows and every statement row that is not a loop keep the stratum of their blocks. The kernel root takes the contract's budget: `kernel.entry_safe: (grid.grid, block.L0_f<K>, block.O0_f<K>) -> kernel.safe` with `K` the `budget` line, and own terminals exist at every stratum.

**Closure.** By induction on the tree: a loop-free statement executes at most its own subtree's functions, at most `N`, stratum 0; a loop of class `c_i` over a body of stratum `k` executes at most `2^(6i) · N · 2^(6k)` node executions; a block of at most four statements of stratum `k` is a loop-free arrangement of at most `N` functions in all, so its statements together are at most `N · 2^(6k)` (each function of the block's own subtree is counted once in `N`, and each nested loop's work is counted in its stratum). So a kernel of root stratum `K` performs at most `N · 2^(6K)` node executions per invocation. The dispatch performs at most `I · N · 2^(6K)`, `I` the contract's largest invocation count. Every loop is bounded (§9.2.7), so every kernel terminates; this is the proof. The **wall bound** is `I · N · 2^(6K) / E_target`, where `E_target` is the target's execution rate, measured once per target as the slowest node family of the kingdom (a global memory load at stride, a subgroup gather, a transcendental) under the worst cache behaviour the harness can arrange, divided by a safety factor of 10; the contract's validation (9.1.2 item 7) requires the wall bound at the chosen `K` to be within the problem's `base_ms`. The harness's watchdog is then an infrastructure fault only: a dispatch that outlives 10 × the wall bound is a device fault, logged with its kernel, never a score, and the run stops until the fault is understood (a kernel that provably executes at most `I · N · 2^(6K)` nodes and still outlives the bound is evidence against `E_target`, which C1's dispatch stage measures over a million kernels: the maximum observed node rate against `E_target`). **OPEN 8:** `E_target` is a measurement with a margin, not a theorem; Codex's rev4 finding 1 stands exactly to that extent, and the design makes the extent explicit and measured.

**Cost.** Five strata multiply the statement-level duals by five (§9.3); a kernel whose nest exceeds `2^(6K)` trips is unspellable at budget `K`; strata round up (a loop of 65 trips costs a `c2` class).

**Removed.** `render::cost`, `worst_bound`, `WORST_D/M/ROWS`, the harness's predicted-time refusal (H6) on an expressed kernel, `scorer.deadline_ms` as a verdict (P3), `gram::wait_gpu`'s runaway as a score (H10: it becomes the infrastructure fault above), `Budget.deadline_factor` as a refusal.

### 9.2.9 Wrap, division, modulo, shifts (layer 3 `index_wraps`; the renderer's division guards and shift masks)

Reasons: `index_wraps`; renderer S9 (shift mask), the `max(b, 1u)` guard and `int_div` helpers; naga's constant-expression overflow.

**Types.** No width ladder. `u32.shift` — a literal shift count, domain `0..=31`. The wrapping duals `u32.indexw`, `u32.uniformw` are gone (A0 deleted their rows).

**Rows.** Integer arithmetic keeps WGSL's defined semantics as the row's meaning: `index.add`, `index.mul` wrap modulo `2^32`; `index.sub_floor` is the floored difference; `index.div: (u32.index, u32.index) -> u32.index` spelled `a / b` with WGSL's defined `x / 0 = x`, and `index.div_nz` spelled `a / max(b, 1u)` as a second row with its own meaning (the reader reads each text as its row, exactly); `index.mod` likewise with `x % 0 = 0` and `index.mod_nz`; `int.div`/`int.mod` spelled through the `int_div`/`int_mod` helpers, whose text is the spelling of WGSL's defined i32 division (`MIN / -1 = MIN`, `x / 0 = x`) on a backend that would otherwise trap — the helper is the row's spelling on that target, not a repair of the tree. Shifts: `index.shl_lit: (u32.index, u32.shift) -> u32.index` spelled `a << k` (`k` a literal below 32); `index.shl: (u32.index, u32.index) -> u32.index` spelled `a << (s & 31u)` as the row's declared meaning; `shr` likewise; `bits.*` and `int.*` shifts the same. Every literal is spelled through the identity helpers (`lit_u32(x)`, `lit_i32(x)`, `lit_f32(x)`, `lit_bool(x)`), as today, so no expression is a constant expression and naga's constant-evaluation overflow (`1u << 40u`, a product of literals) cannot occur.

**Closure.** Wrap is WGSL-defined arithmetic and reaches no index of a store (§9.2.3: store indices are the own templates, proved in range), no array index (§9.2.2: masked), and no loop bound beyond making it smaller (§9.2.7). A wrap elsewhere is a wrong value, which fitness scores. Division by zero and shifts by any amount are defined by the rows' texts. **OPEN 9:** the i32 division helper's necessity on naga 30's Metal backend (naga may already spell WGSL's division semantics) is a target probe of §9.1.2 item 8; whichever text the probe shows correct is the row's one spelling.

**Cost.** None in expressiveness; `index_wraps` stops being a fault.

**Removed.** The oracle's layer 3 `index_wraps` on the path (the oracle keeps it as a diagnostic only, `Site::Other`), the renderer's `max(b, 1u)` insertion as a step separate from the row's text, S9 as a step (the mask is the row's text).

### 9.2.10 Literals and knob references (layer 1 literal reasons, `knob_reference_unresolved`)

Reasons: `nonfinite_literal`, `literal_not_a_row`, `literal_with_children`, `knob_reference_unresolved`; renderer S3 (literal snapping), R3 (non-finite literal, a literal with no text).

**Types.** Every constant-capable dual's `C_T` domain (§4) is its `literal` column: f32 duals the finite f32 range, `u32.blk` `{1, 2, 4, 8}`, `u32.lit64` `1..=64`, `u32.code4` `0..=3`, `grid.shape` the shape table, `u32.passes` the contract's range, `u32.payload` `1..=2^20`, `u32.shift` `0..=31`. `u32.boundm` and the other binding bounds have no literal (they are terminals).

**Rows.** `literal.<form>: -> <dual>` is the placeholder `?_T` of its dual; a `Dc_T` word names a value of the domain, and Dc mutation, recombination and transposition select valid `C_T` indices (§4). Knob references `literal.p_<knob>: -> u32.index | u32.lit64 | u32.lit4096` carry a `u32.payload`-domain constant and spell the value `clamp(knob · payload / 256, lo, hi)` of the output dual's domain, computed from the kernel's own grid node at render time; the knobs `wcol, wrow, bcol, brow, colspan, rowspan, threads, ktile` all exist in every kernel because `ktile` is a component of every grid (`grid.with_ktile` becomes the fourth argument of `grid.of`, default `1`).

**Closure.** A literal's value is a member of its domain by construction of `C_T`; the walk expresses `?_T` only as a leaf; every knob a reference names exists in every kernel. `literal_with_children` cannot occur because a literal row has no signature rows.

**Cost.** None.

**Removed.** `render::snap`, `lit` clamping (S3), `params::resolve` as a fallible step (it is a total function of the grid node), `literal.p_*` over `u32.wglen` (no wglen).

### 9.2.11 Render and parse (layer 7 `kernel_does_not_render`, `kernel_does_not_parse`; R1–R4)

**Types and rows.** None new. The renderer is a total function `render: Tree -> String` (§9.4): every row of every generated table has a spelling; the generator's test renders every row once at an instance of each input (§9.6.3) and parses the text in naga with the target's capabilities. The `load.cfg_*` rows spell the interface's own field names (A0 fixed `pcfg.d`).

**Closure.** Every node is a row with a spelling (frame membership), so `render` has no `Err` branch to take; `kernel_does_not_render` and `kernel_does_not_parse` are generator defects caught by §9.6.3, not properties of a gene. R2 is the rung ladder (§9.2.6), R3 the domains and the target (§9.2.1, §9.2.10), R4 the per-entry alphabet (an interface's rows are the only rows; a workgroup array is consumed only by access rows; every helper has a scalar by the generator's test).

**Removed.** Every `Err` return of `render.rs` on an expressed tree (A1 R1–R4), `Rendered.masked`, `Checker::safety`'s render-and-parse on the path.

### 9.2.12 Atomics and determinism

Reasons: A3 7.6 (determinism), Codex rev4 finding 2, `statement_not_in_kingdom` (ray queries, image atomics, cooperative stores: no rows).

**Types.** `u32.atomicold`, `i32.atomicold` — "the old value an atomic read-modify-write returned: order-dependent across invocations".

**Rows.** Workgroup and arena read-modify-writes output the `atomicold` duals: `atomic.add_u: (wgmem<atomic<u32>,2^j>.index, u32.wgidx_<j>, u32.index) -> u32.atomicold`; `atomic.<a>_add: (u32.aidx_<a>, u32.index) -> u32.atomicold` for an arena. Consumers in `determinism strict`: `stmt.discard_u_<rung>: u32.atomicold -> stmt.<rung>` (`_ = e;`) only, and `atomic.load_*` after a phase barrier reads the deterministic final value of a commutative reduction (`add`, `sub`, `min`, `max`, `and`, `or`, `xor`); `atomic.exchange_*` is absent. In `determinism relaxed` (the V24 contracts, whose designs allocate slots by `atomicAdd`): the conversion rows `index.of_atomicold: u32.atomicold -> u32.index`, `int.of_atomicold`, and `exchange`. The harness's output kind states what fitness compares (V24: pooled recall, order-free).

**Closure.** In a strict kingdom no row carries an order-dependent value into an index, a store or a bound, so every stored value is a deterministic function of the inputs; in a relaxed kingdom the dependence is declared by the contract and the fitness measure is order-free by the problem's declaration.

**Cost.** Strict kingdoms spell histograms and reductions through atomics with their results read after a barrier; slot allocation needs a relaxed contract.

**Removed.** `atomic.*_u`/`_i` outputting `u32.index`/`i32.int` directly; `atomic.exchange` from strict kingdoms.

### 9.2.13 The typed tree and the root (layers 1, 2 root; `module_entries_not_the_interfaces`; `no_own_terminal`)

Reasons: `unknown_dual`, `unknown_row`, `signature_mismatch`, `no_own_terminal`, `root_not_safe_kernel`, `module_entries_not_the_interfaces`, `grid_not_in_grammar`.

**Types and rows.** The root of a gene of `K_e` is `kernel.safe`, produced only by `kernel.entry_safe: (grid.grid, block.L0_f<K>, block.O0_f<K>) -> kernel.safe` when the contract names a `primary` output (the epilogue's last statement is a store to `u32.own_<primary>`), else `kernel.entry_safe: (grid.grid, block.L0_f<K>) -> kernel.safe`. A module is the chromosome's fixed linker (§9.1.4). Every demanded dual has a terminal (`docs/PLAN_wgsl_terminals.md`, delivered 034bf41f); the `own.*` rows are real terminals with their own text. The tables' fallback columns (`fallback`, `fallback_bits`, `fallback_noop`, `fallback_elem`) are removed (A0).

**Closure.** The kingdom check of §1 and the walk of §3: every node is a row at its signature, every tail cell a terminal of its type, the root a producer of the root type. These reasons are the kingdom check's, run once per table, never per gene.

**Cost.** None. **OPEN 10:** for an entry with no `primary` output (V24 `ingest` writes arenas only) the "every gene writes its output" purpose of the epilogue is carried instead by fitness; whether an arena store should be demanded as the epilogue's last statement is a problem choice.

## 9.3 Layout bounds for one V24 kingdom at `N = 8`

The bounds are those of §2 and `KingdomContract::bounds`; the build measures them in B1 by the kingdom check (`the_kingdom_reports_print`); this section gives the derivation and an estimate so the design is seen to be buildable.

**Type count of `K<M3 Max, v24_ingest5b, search, contract, strict aliases, policy rows cands 64 + arena index_u32 index_f32, determinism relaxed>`.**

| family | duals today | duals in `K` | why |
|---|---|---|---|
| value duals (scalar, vector, matrix, register, array families) | 158 | about 165 | − partition origins (8), − `wgindex`-style (7), − `mirror`, `cell`, `phi`, `wglen`, `decl`, `decls`, `payload` kept; + `own_cands`, `ownrow_cands`, `aidx_index_u32`, `aidx_index_f32`, `wgidx_<j>` × 4 classes of the map, `wgown_<j>` × 4, `pidx_<j>` × 3, `xcounter<d>` × 6, `atomicold` × 2, `shift`, `boundx<c>` × 4, `boundu<c>` × 4, `passes`, `shape` |
| W-value family (§9.2.5, OPEN 5) | 0 | about 60 | the scalar, register and private-array duals a W statement's values range over |
| statement-level duals | 44 | 61 × 5 strata = 305 | L, C (12 each), O (10), B (5), W (12), WB (5), kernel, grid; × fuel |
| **total** | **202** | **about 530** | |

**Arities.** `n_U` is 4 for every statement and block type (`seq4`, `row_plus_g`, `grid.of`), 3 for `u32.index` (`index.clamp`), 2 or 3 for the rest; `n_max = 4` as today. `a_T` for the widest demands: `a_{stmt.L<d>_f<k>} = 4` (`seq4`), `a_{u32.index} = 3`, `a_{f32.real} = 3` (`fma`, `clamp`), `a_{u32.uniform} = 3` (`uniform.clamp`), `a_{block.*} = 2` (`if_else`, phase), `a_{u32.own_cands} = 1`.

**Heads at `N = 8` (rev 4 §2.1, `h_T = [T = R] + 7 · a_T` over the reachable types).** `h_root = 1`; every statement and block dual of stratum ≤ K is reachable from the root through the lift and sequence rows, so `h = 28` for each `stmt.*` dual and `h = 14` for each `block.*` dual; `h = 21` for `u32.index`, `f32.real`, `u32.uniform`; `h = 7` for a dual demanded once per function (`u32.own_cands`, `bool.uflag`, the origins); `h = 0` for a dual no function demands (`grid.shape` is demanded by `grid.of`: `h = 7`).

**Width.** Today's measured table (rev 4, 2026-10-20): 202 types, `L = 3,958`, width 58,090 words, the widest type `h 21, P 1,379, t 1,358, d 1,379`. The width is a sum over types of `h_T + t_T + d_T` with `t_T ≈ P_T − h_T` and `P_T = Σ_U h_U · a_{T,U}`, so it scales with the number of reachable producer types times their head lengths. With about 530 reachable types, of which 305 statement-level at `h = 28` or `14`, the estimate is **about 150,000 words a gene at `N = 8`** (2.5 × today's), and about 650,000 at `N = 32`. A million genes at `N = 8` is then about 600 GB through the paged geneframe for C1, so C1 streams in pages as rev 4 says; the levers, if the measured figure must come down, are the fuel strata (5 → 3 costs one factor of trip-class resolution) and the W-value family (OPEN 5). **OPEN 11:** the measured width from B1 and whether it is acceptable for C1's million-gene closure test or C1 runs at `N = 4` for the shared-core operators and `N = 8` for the initialiser.

**Terminals per demanded type.** Every value dual has a `?_T` or an `own.*`; `grid.shape` has `?_shape`; every statement-level dual at every stratum has `own.stmt_*` (an empty statement) and `own.block_*` (an empty block, or for O rungs a store of the thread's first cell of the primary output); `u32.own_cands` has `own.u32_own_cands`.

## 9.4 The renderer as a pure function of the tree

`render(tree, C) -> String` is total on expressed trees. It spells: the prologue from `C` (directives, structs, bindings, the workgroup map, the helper library, the register and counter declarations), the `// grid`, `// mode`, `// passes` lines from the grid node and `C`, then the tree, post-order, each node by its row's spelling template with `{0}`, `{1}`, ... for its children. No node is looked up for a reason to do otherwise. The spelling of each family, with every mask, clamp and guard as the row's own declared meaning:

| family | spelling (the row's meaning) |
|---|---|
| `literal.*`, `const.*` | the `C_T` value through the identity helper of its scalar: `lit_u32(5u)`, `lit_f32(0.5f)`, `lit_i32(-3i)`, `lit_bool(true)`; vectors and matrices splat the one value |
| `literal.p_<knob>` | `lit_u32(<clamp(knob · payload / 256, lo, hi)>)`, the value computed from the kernel's grid node |
| `own.*` of a value dual | the declared value: `lit_u32(0u)`, `false`, the thread's origin (`gid.x * BCOL`), the thread's first cell (`cell_b(origin_row, origin_col)`) |
| `own.stmt_*`, `own.block_*` | `{}`; for `O<d>`: `{ store_primary(own cell, lit(0)); }` |
| `builtin.*`, `axis.*` | the builtin's name and component: `gid.x`, `lidx` |
| `arith.*`, `trig.*`, `geom.*`, `int.*`, `bits.*`, `compare.*`, `logic.*`, `select.*`, `shape.*`, `convert.*`, `quantise.*`, `hash.*`, `fixed.*` | the naga operator or builtin, or the composite helper's call, over the children; `index.shl`: `({0} << (({1}) & 31u))`; `index.div`: `({0} / {1})`; `index.div_nz`: `({0} / max({1}, 1u))`; `int.div`: `int_div({0}, {1})` (§9.2.9) |
| `safe.wg_of_<j>`, `safe.pidx_<j>` | `(({0}) & (2^j − 1)u)` |
| `safe.wg_own_<j>` | `min(lidx + ({0}) * THREADS, (2^j − 1)u)` |
| `safe.thread_row`, `safe.thread_col` | the origin under the contract's partition (§9.2.3) |
| `safe.row_plus*`, `safe.col_plus*` | `({0} + min({1}, BROW − 1u))` and the strided and grouped forms |
| `safe.cell_<b>` | `({0} * S_b + {1})`, or `{0}` for a 1-D output |
| `safe.ownrow_<r>` | `(wid.x * W + min({0}, W − 1u))` |
| `arena.idx_<a>` | `(({0}) & (L − 1)u)` or `(({0}) % Lu)` |
| `load.<b>` (input) | `b[{0}]`; `load.<b>_own`: `b[{0}]`; `load.<a>` (arena): `a[{0}]` or `atomicLoad(&a[{0}])` |
| `store.<b>_<rung>` | `if (G_b) { b[{0}] = {1}; }` with `G_b` the generator's guard (empty when none is needed); `store.<b>_add_*`: `b[{0}] = b[{0}] + {1};`; `store.<b>_mirror_*`: `if (r < c && G_b) { ... }` |
| `wg.get_<e>_<j>`, `wg.get_own_<e>_<j>` | `a[{1}]`; lane forms `a[{1}][({2}) & 3u]` |
| `wg.set_<e>_<j>_W<d>` | `a[{1}] = {2};` |
| `wg.phase_L<d>`, `wg.phase_s_L<d>` | `workgroupBarrier(); {0} workgroupBarrier(); {1}` / `storageBarrier(); workgroupBarrier(); {0} storageBarrier(); workgroupBarrier(); {1}` |
| `wg.put_*`, `wg.reduce_*`, `wg.uload_*` | as today's text (barriers inside) |
| `atomic.*` | `atomicOp(&a[{1}], {2})`; `stmt.discard_*`: `_ = {0};` |
| `subgroup.<op>_<reg>_L<d>` | `{0} = subgroupOp({1});`; `stmt.subgroup_barrier_*`: `subgroupBarrier();` |
| `reg.get_*`, `reg.set_*_<rung>`, `parr.*` | the register's name; `r = v;`; `pa[{1}] = {2};` |
| `block.one/seq2/seq3/seq4_*` | the statements in order; `stmt.block_*`: `{ {0} }`; `stmt.lift_*`: `{0}` |
| `stmt.if_*`, `stmt.if_else_*`, `stmt.if_uflag_*`, `stmt.if_u{le,eq,ne,lt}_*` | `if ({0}) { {1} }`, `if ({0}) { {1} } else { {2} }`, `if ({0} <= {1}) { {2} }` |
| `block.brk_B<d>` | `{0} if ({1}) { break; } {2}`; `block.cont_*`: `if ({0}) { continue; } {1}` |
| `loop.*` | `let n<d> = <bound>; for (var k<d> = 0u; k<d> < n<d>; k<d> = k<d> + s) { body }`; `items_cont_*` without the reset; `unroll_*` the body `n` times with the counter a literal in each copy |
| `bound.of_index_<c>`, `bound.of_uniform_<c>` | `min({0}, 2^c u)`; `bound.m`: `cfg.m` |
| `safe.tile_counter<d>`, `safe.xtile_counter<d>`, `safe.counter` | the counter variable's name (`k<d>`, `kx<d>`, `t`) |
| `uniform.*`, `uflag.*`, `index.of_uniform` | the same text as the index row or field they stand for |
| `grid.of`, `grid.of_rows_x`, `grid.with_passes` | `@compute @workgroup_size(x, y, 1)` from the shape, the `// grid x y bx by[ rows-x]` line, `// passes N` |
| `kernel.entry_safe` | the entry function: declarations, `{1}`, `{2}` |
| `kernel.module_of_<I>` (the chromosome's linker) | the prologue once, then each gene's entry function in the interface's order |

Rules of the function: (1) the spelling of a row never depends on where the node sits or on any other node (the counter variables and knobs are kernel-scope names); (2) the generator's test spells every row once and parses the result (§9.6.3); (3) `render` returns `String`, never `Result`, and records nothing (no `masked`, no events).

## 9.5 Reading (C4) under this section

The reader (`safe_reader.rs`) maps a kernel's text onto the rows of its contract's `K_e` exactly: each text form has one row (`a / b` is `index.div`, `a / max(b, 1u)` is `index.div_nz`, `a << (s & 31u)` is `index.shl`, `a << 3u` is `index.shl_lit`). A text that spells something `K_e` has no row for is outside the kingdom by a named reason (a store at a non-template index, a workgroup write outside a phase, a barrier under a thread condition, a loop whose bound its body changes, a workgroup array not in the map, a shape not in the table, a trip product past the budget), listed as C4 says. The reader performs no rewrite (`lower_returns_in_loops`, override defaults, `conform`, zero substitution are gone, R1).

## 9.6 Conformance: the C1 closure test and the generator's tests

### 9.6.1 C1, as the specification states it

For every `K_e` of every contract in use (each V24 entry point, the Gram, nluca, kmeans): on the GPU, under the lock, 1,000,000 genes from the typed initialiser of the shared core and 1,000,000 children per shared-core operator (point mutation, inversion, IS, RIS, gene transposition, one-point, two-point and gene recombination, Dc operators), each from parents drawn by the initialiser. The stages are instrumented separately and each asserts:

1. **generation**: no call into `check.rs`, `safety.rs`, `check_legal_under`, `render::cost`, `kernel_info` or any legality function (a link-time test: the evolution binary's symbol table has none of them; and a run-time hook that counts calls, asserting zero);
2. **expression**: the device walk and the host `express` agree node for node on a 1 % sample; every node's type is its slot's type; no `LayoutDefect`;
3. **rendering**: `render` returns a string for every tree (it cannot fail by type); zero events (there is no event channel to count: the assertion is that the renderer's type is `Tree -> String`); the text parses in naga and validates under the target's capabilities and limits (`Target::validate`): zero failures;
4. **dispatch**: a compute pipeline is created on the target for every rendered module (a creation failure is a kingdom bug of the target, reported with the row family that caused it); a 1 % sample is dispatched at the contract's smallest shape with an iteration counter buffer: the measured node executions are at most `N · 2^(6K)` per invocation, and the measured per-node rate is within `E_target`;
5. **oracle**, afterwards and only here: `check.rs` layers 1–7 and `safety.rs` under the contract, on every rendered kernel: zero failures; a capacity verdict (`too_large_to_prove`, `STEP_CAP`) is an oracle defect to fix (raise the cap or make the oracle incremental), never a pass.

Before launch the report states the oracle's measured cost per kernel and the run's expected time; the paged geneframe streams the genes (about 600 GB at `N = 8` by §9.3). The test is the gate of B2's sub-steps (for the layers each closes) and of C1.

### 9.6.2 The generator's own tests, per instantiated kingdom

1. **Ownership enumeration.** For every written non-arena binding, every template and every value of the block factors: enumerate every invocation's owned cells at the contract's smallest dispatch and smallest params, and at the largest params with the smallest dispatch; assert pairwise disjointness and that `r * S_b + c` is within `u32` and within the length polynomial under `G_b`. (Today's `safety.rs` exhaustive small-domain test, moved to the generator.)
2. **Map budget and shapes.** The workgroup map's bytes within `max_workgroup_storage`; every shape of the table within the limits; `own.grid_grid` legal.
3. **Every row spells and parses.** Each row rendered once with own terminals for its inputs, inside a minimal kernel, parsed and validated by naga under the target; every helper present; every composite has a body.
4. **Feature probes.** One kernel per feature family compiled and dispatched on the target before the family's rows are emitted.
5. **Kingdom check and bounds.** `KingdomRelations::check` passes; the bound table at the layout's `N` is printed to the instantiation report (the measured width of §9.3).
6. **Fuel.** For every loop row, the output stratum equals the body's stratum plus the trip class; no row outputs a stratum above 4; the root takes the contract's `K`.
7. **Phases.** No row outputting `stmt.W<d>` or `block.W<d>` spells a barrier or a read at `u32.wgidx_<j>`; the only rows spelling a workgroup write output `stmt.W<d>`; the only rows spelling a barrier output `stmt.L<d>`.

### 9.6.3 What C1 does not test

Fitness quality (C5), seed coverage (C4, S1), and `E_target`'s truth beyond the dispatch-stage measurement (OPEN 8).

## 9.7 Open questions

1. (§9.2.1) The workgroup shape table: all 7,690 legal `(x, y)` of the target as a literal domain, or the 66 power-of-two pairs; S1's seeds decide.
2. (§9.2.2) The fixed workgroup-memory map versus a typed buddy-tree of declarations (about 60 more duals, evolvable lengths).
3. (§9.2.2) A read past a read-only binding returns its last element as the row's meaning; or input reads at a proved index too (costs free `phi[t*d + c]` addressing).
4. (§9.2.3) The `writes` and `partition` facts of the V24 contracts per entry must come from the problem (does `ingest` write `cands`?).
5. (§9.2.5) The W-phase value family: a second value family (about 60 duals, counted in §9.3) or a restricted W statement list; B1 measures both widths and takes the smaller that passes §9.6.2 item 7.
6. (§9.2.5) A bare `workgroupBarrier();` statement as a phase with an empty write block, for seeds that barrier without writing.
7. (§9.2.6) A counter read outside its loop: the variable's value (0 before, final after), the pure spelling adopted here; today's renderer writes `0u`.
8. (§9.2.8) `E_target` is a measurement with a tenfold margin, verified by C1's dispatch stage, not a theorem; a dispatch outliving ten times the wall bound is an infrastructure fault that stops the run.
9. (§9.2.9) Whether naga 30's Metal backend already spells WGSL's defined i32 division; the probe decides the one spelling.
10. (§9.2.13) An entry with no `primary` output has no epilogue; whether an arena store should be demanded as the last statement is a problem choice.
11. (§9.3) The measured width (B1) and whether C1 runs its million genes at `N = 8` or the operators at a smaller `N`.
12. (§9.2.4) Row-owned outputs leave intra-workgroup collisions to the design, as today's rows mode does; whether V24's `cands` should instead be owned per invocation (`lidx < 64` lanes of the row) is a contract choice the seeds decide.

## 9.8 Index: every ledger rule, reason and family to its section

| rule (plan ledger) | reasons / items | section |
|---|---|---|
| contract, target limits, features (layer 5) | `workgroup_size_over_limit`, `workgroup_memory_over_limit`, `feature_missing`; R3 | 9.2.1, 9.2.2 |
| array range (layer 3) | `index_out_of_range`; S7; R4 (array as value) | 9.2.2 |
| write ownership, disjointness (layer 7) | `write_sets_overlap`, `store_index_not_affine`, `store_index_may_wrap`, `store_stride_may_be_zero`, `index_not_in_bounds`; `store_cell_wraps`; S4, S8 | 9.2.3 |
| output reads, aliasing, read-after-write (layer 7) | `output_read_of_another_invocations_cell`; `alias` partition | 9.2.3, 9.1.2 item 3 |
| workgroup races and ordering (layer 7) | `workgroup_memory_race`; S10 | 9.2.5 |
| barrier and subgroup uniformity (layer 4, layer 7) | `barrier_in_nonuniform_control_flow`, `barrier_in_nonuniform_control`, `subgroup_operation_in_nonuniform_control` | 9.2.6 |
| bounded loops and work budget (layer 6, Metal hang) | `loop_bound_not_uniform`, `loop_bound_changes_in_body`, `loop_bound_wraps`; S6; P3, H6, H10, the cost estimator | 9.2.7, 9.2.8 |
| wrap (layer 3 `index_wraps`) | `index_wraps` | 9.2.9 |
| division, modulo, shifts | the `max(b, 1u)` guard, `int_div`, S9 | 9.2.9 |
| atomics | determinism (Codex rev4 finding 2), `statement_not_in_kingdom` | 9.2.12 |
| typed tree, grammar (layers 1, 2) | `unknown_dual`, `unknown_row`, `signature_mismatch`, `no_own_terminal`, `root_not_safe_kernel`, `break_outside_loop`, `continue_outside_loop`, `block_not_in_safe_grammar`, `statement_not_in_safe_grammar`, `barrier_under_condition`, `loop_not_bounded_constructor`, `nested_item_loop`, `tile_depth_exceeded`; R2 | 9.2.13, 9.2.6 |
| render and parse (layer 7) | `kernel_does_not_render`, `kernel_does_not_parse`, `interface`, `parse`; R1 | 9.2.11 |
| harness kernel facts (H1) | `// mode`, block factors, `// grid`, binding schema; H2 | 9.2.1, 9.1.4 |
| `memory_barrier_without_control` | | 9.2.5 |
| knob reference, duplicate decls, grid, workgroup array grammar | `knob_reference_unresolved`, `grid_not_in_grammar`, `workgroup_array_not_in_grammar` | 9.2.10, 9.2.1, 9.2.2 |
| literals | `nonfinite_literal`, `literal_not_a_row`, `literal_with_children`; S1, S3 | 9.2.10, 9.2.1 |
| arena index, layer 7 policy | `arena_index_not_in_bounds`, `--layer7` | 9.2.4 |
| `module_entries_not_the_interfaces` | | 9.1.4, 9.2.13 |
| execution budget, device deadline | P3, H6, H10 | 9.2.8 |
| renderer rewrites (ten families) | S1–S10 | 9.2.1 (S1, S2), 9.2.10 (S3), 9.2.3 (S4, S8), 9.2.6 (S5), 9.2.7 (S6), 9.2.2 (S7), 9.2.9 (S9), 9.2.5 (S10) |
| oracle capacity | `too_large_to_prove` | 9.6.1 item 5 |

All 21 ledger rows, the 47 `check.rs` and `safety.rs` reasons of A1 §7 (7 + 13 + 3 + 1 + 3 + 3 + 17 by layer), H1's four facts and H2, the ten renderer families and R1–R4, and the execution budget are assigned.

## §9.7 decisions (lead, 2026-10-21)

Each open question is decided toward the goal (every legal kernel spelled; every expressed tree legal; no rewrite):

1. Shape table: the target's full legal domain (all legal (x, y)), so every legal seed reads (C4).
2. Workgroup memory: the typed buddy-tree of declarations (evolvable lengths); B1 reports its width.
3. Read past a read-only binding: the row's declared meaning is the device's defined clamp to the last element (wgpu's Restrict policy), spelled in the row's text; free addressing is kept.
4. Per-entry `writes` and `partition` facts: declared by the problem definition (`problem.toml`); B1 adds them for V24 from the problem owner's code.
5. W-phase values: both forms built in B1 and their widths reported; the narrower one that spells every seed is kept.
6. A bare `workgroupBarrier();` is a phase with an empty write block.
7. A loop counter read outside its loop spells the variable's real value (0 before the loop, the final value after); the `0u` rewrite is removed.
8. E_target is measured per target with a tenfold margin and verified by C1's dispatch stage.
9. i32 division: a probe on naga 30's Metal backend picks the single spelling.
10. An entry's epilogue store is a problem fact.
11. B1 prints the measured bound table; C1's size is set from it.
12. Output ownership is per invocation (lane ownership), so no two invocations of a workgroup can write one cell.
