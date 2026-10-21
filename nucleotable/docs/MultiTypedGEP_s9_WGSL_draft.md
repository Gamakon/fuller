# 9. WGSL legality as types (draft for revision 5)

**Revision 2 (2026-10-21), after the advisor's review of the draft with the lead's decisions (`docs/reviews/2026-10-21_true_gep_spec_advisor.md`, FAIL, nine findings on this file).** What changed:

1. (finding 1) `safe.wg_own_<j>` no longer clamps: every workgroup write and own read spells the guard `if (i < len)` with `i = lidx + k * THREADS` unclamped, so the write set is injective by the row's text; §9.6.2 item 7 asserts injectivity over `lidx` for every `k` (§9.2.2, §9.2.5, §9.4).
2. (finding 2, Codex rev4 finding 2) strict atomics: a workgroup `atomicLoad` is an L-rung statement bracketed by barriers into a register (`atomic.uload_*`); an arena `atomicLoad` outputs `u32.atomicold`; plain-valued atomic loads exist only under `determinism relaxed`; atomic read-modify-writes are rows of the kinds where they are race-free and order-independent (§9.2.12).
3. (finding 3) decision 2 replaced: the workgroup-memory map is a fixed **stratum map**, one array per power-of-two stratum per element type, total within `max_workgroup_storage`, declared from the map in the prologue; a use chooses a stratum (§9.2.2). The buddy-tree of gene-carried declarations is withdrawn as unsound (a declaration subtree cannot bind uses elsewhere in the gene).
4. (finding 4) the store row takes `(u32.threadrow, u32.threadcol, elem)` and spells the cell and the guard itself; `u32.own_<b>` remains for `load.<b>_own` only; the mirror store likewise (§9.2.3, §9.4).
5. (finding 5) decision 12 replaced: `policy rows` is gone; `partition lane <W>` is a fourth partition value with origin `wid.x * W + lidx`, block 1 and the guard `lidx < W` in the store row's text; §9.2.4 keeps arenas only; S1 reports which `cands` writers fall outside (§9.2.3, §9.2.4).
6. (finding 6) decision 3 replaced: a read of a read-only binding spells its clamp in the row's text, `b[min(i, arrayLength(&b) - 1u)]`, target-independent; the reader reads a bare `b[i]` as outside the kingdom (§9.2.2, §9.5).
7. (finding 7) decision 7 fixed: a loop spells `k<d> = 0u; for (; k<d> < n<d>; k<d> += s)` so the kernel-scope counter is not shadowed; after an unrolled loop a read is `0u` because an unrolled loop never assigns it (§9.2.6, §9.2.7, §9.4).
8. (finding 8) a knob reference spells a WGSL expression over kernel-scope `const`s the prologue emits from the grid node (`KNOB_WCOL`, ...), never a value computed from another node (§9.2.10, §9.4).
9. (finding 9) decision 5's restricted-list form needs a transitive test: §9.6.2 item 7 checks, over the reachability closure of every value dual a W statement can demand, that no producer reads a foreign element.
10. (Q2) the fuel induction states the function-count argument the block case uses (§9.2.8).
11. §9.7 is now the decided list; the lead's decisions 2, 3, 5, 7, 12 are replaced by the forms above; §9.8 index updated. Finding 10 (plan text lagging the spec: ledger rows 42, 43, 53, plan decision 2, the duplicated C2 sentence) is in `docs/PLAN_true_gep_wgsl.md`, not this file, and is left to the lead.

Draft of §9 of `MultiTypedGEP.md`, task D1 of `docs/PLAN_true_gep_wgsl.md` (revision 4), written 2026-10-21 against `docs/GOAL.md`, rev 4 §1–§7, the rule ledger, the five reviews, the A1 and A3 reports, `kingdoms/safewgsl/gen_tables.py` (post-A0 working tree), `src/wgsl/{check,safety,render,harness,safe_reader}.rs`, `src/typed_gep.rs` and `src/kingdom_contract.rs`. It is the specification the build (B1, B2, B3, C1, C2) is held to. Revision 1's open questions are decided in §9.7; the two that remain measurements rather than theorems (`E_target`, the i32 division probe) are marked **OPEN** in the text.

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
| `partition contiguous|strided|grouped <G>|lane <W>` | how an invocation's block of an output is laid out (§9.2.3); `lane W`: one cell of the workgroup's row of `W` cells per invocation, `lidx < W` | the nine `cell_of` pairs; today's `rows` mode |
| `mode <m>...` | the modes the problem lists; `K` is instantiated per listed mode (`K<..., mode>`) | `KernelSpec.modes`, A1 H1 |
| `passes lo hi` | the `// passes N` range when the problem reads it | `KernelSpec.passes_header` |
| `policy strict|arena <b>...` | layer 7's policy, named: which bindings are arenas (the design's own state, data-dependent indices); every other written binding is owned by partition | `--layer7`, `Contract.arena`; `rows` mode is now `partition lane` |
| `determinism strict|relaxed` | whether an order-dependent atomic value (a returned old value, an in-kernel arena `atomicLoad`) may be consumed (§9.2.12) | none (Codex rev4 finding 2) |
| `wgmap <name>` | the workgroup-memory **stratum map**: for each element type, one array per power-of-two stratum `2^j` elements the map lists, total within the target (§9.2.2) | `wg.decls` knobs, the renderer's halving |
| `budget <K>` | the root fuel stratum `K` in `0..=4` (§9.2.8): work per invocation at most `N · 2^(6K)` node executions | none (the deadline) |

### 9.1.2 Validation on the host, before any gene exists

Instantiation runs `validate(C)` once. It is input validation of a human-written file; it never sees a gene. Every defect is "our software has a bug: ..." and stops the instantiation. It establishes:

1. every `param` range is non-empty and within `u32`; every uniform field of the interface has a `param` line; every `arrayLength(b)` of a written binding has a `length` polynomial that is positive over the ranges;
2. `workgroups` per axis within the target's dispatch limits; the target's limits positive;
3. the alias partition is a partition: two bindings in one class are one buffer; a class with a written member has no two members whose ranges overlap unless the contract declares them identical (same offset and length); every binding not named is its own class;
4. `writes` names read_write bindings only; `primary` is one of `writes` or absent; arena bindings are read_write; every written binding that is not an arena has a `partition` fact, and a `lane W` partition has `W ≤` the workgroup's invocation count for every shape of the table that the entry may use (else the store's guard `lidx < W` leaves cells unwritten, which the report says);
5. every `mode` listed is one the harness lists for the problem; `passes` within `1..=16`;
6. the `wgmap`'s arrays (one per listed stratum per element type), each rounded up to 16 bytes as Metal allocates them, total at most `max_workgroup_storage`;
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

**Cost.** One kingdom per `(target, mode)`; the shape table is the target's full legal domain (lead's decision 1), so every legal seed's shape reads and nothing is lost there.

**Removed.** `literal.wgdim`, `u32.wgdim`, `workgroup_dim`, `workgroup_shape`, `snap` on block factors (renderer S1, S2), `Target::check_limits` on expressed kernels (oracle only), `features()` of layer 5 (oracle only), the mirror rows from every `K` whose mode is not `full` (§9.2.3).

### 9.2.2 Array range: workgroup, private, storage reads (layer 3, `index_out_of_range`; the `wg.decls` grammar)

Reasons: `index_out_of_range`, `workgroup_array_not_in_grammar`, duplicate `decls`, `index_not_in_bounds` for reads; renderer S7 (array re-masking), R4 (a workgroup array used as a value).

**Types.** The workgroup-memory **stratum map** `wgmap` of the contract fixes, for each element type `e` in `{f32, vec2<f32>, vec4<f32>, u32, i32, atomic<u32>, atomic<i32>}`, the strata `j` it lists, and for each listed `(e, j)` exactly one array `wg_<e>_<j>` of `2^j` elements, all declared in the prologue from the map; their total is within `max_workgroup_storage` by validation (9.1.2 item 6). An array is a terminal of a dual that carries its element and stratum: `wgmem<f32,2048>.real` (the array `wg_f32_11`), `wgmem<vec4<f32>,512>.real`, `wgmem<u32,1024>.index`, `wgmem<i32,1024>.int`, `wgmem<atomic<u32>,256>.index`, `wgmem<atomic<i32>,256>.int`; a kernel's use of an array is a use of its stratum, and "evolvable lengths" means a mutation moving a use from one stratum's array to another's. One index dual per stratum: `u32.wgidx_<j>` ("an index below `2^j`, by the mask its row writes") and `u32.wgown_<j>` ("the unclamped candidate `lidx + k · THREADS` of this invocation's own element; the access row guards it below `2^j`"). Private arrays likewise, one array per listed stratum per element type: `parr<f32,64>.real`, `parr<vec4<f32>,16>.real`, `parr<u32,16>.index`, `parr<u32,64>.index`, `parr<u32,128>.index`, `parr<bool,16>.flag`, `parr<vec2<f32>,16>.real`, with `u32.pidx_<j>`.

**Rows.** `safe.wg_of_<j>: u32.index -> u32.wgidx_<j>` spelled `(i & (2^j - 1)u)`; `safe.pidx_<j>: u32.index -> u32.pidx_<j>` spelled the same; `safe.wg_own_<j>: u32.lit64 -> u32.wgown_<j>` spelled `(lidx + k * THREADS)` and `safe.wg_own_tile_<j>: u32.tcounter<d> -> u32.wgown_<j>` likewise (`THREADS` is the kernel-scope knob constant of §9.2.10; no clamp: a clamp would send two invocations to one element, the advisor's finding 1). Every foreign read of an array of stratum `j` takes `u32.wgidx_<j>`: `wg.get_<e>_<j>: (wgmem<e,2^j>, u32.wgidx_<j>) -> e` spelled `a[i]`. Every own access takes `u32.wgown_<j>` and spells the guard: the write `wg.set_<e>_<j>_W<d>: (wgmem<e,2^j>, u32.wgown_<j>, e) -> stmt.W<d>` is `if (i < 2^j u) { a[i] = v; }`, the own read `wg.get_own_<e>_<j>: (wgmem<e,2^j>, u32.wgown_<j>) -> e` is `select(ZERO_e, a[min(i, 2^j - 1u)], i < 2^j u)` (the element where it is owned, the element type's zero where the candidate is past the array; the `min` keeps the read in range, the `select` keeps the value the row declares). Private arrays: `parr.get_<e>_<j>: (parr<e,2^j>, u32.pidx_<j>) -> e`, `parr.set_<e>_<j>_<rung>: (parr<e,2^j>, u32.pidx_<j>, e) -> stmt.<rung>`. Lane accesses (`get_v_lane`, `set_v_lane`) spell `[(j) & 3u]` as their declared text. A gene has no declaration rows.

Storage reads: for every storage binding `b` of the interface that is not written by this entry, `load.<b>: u32.index -> <elem>` spelled `b[min(i, arrayLength(&b) - 1u)]`: the clamp to the last element is the row's own text, so its meaning is the same on every target whatever bounds policy wgpu installs (naga `Restrict` on Metal, another on Vulkan). Reads of an output binding are at owned cells only (§9.2.3); reads of an arena at arena indices (§9.2.4).

**Closure.** An array of stratum `j` is accessed only by rows whose index input is `u32.wgidx_<j>` / `u32.wgown_<j>` / `u32.pidx_<j>`; a `wgidx_<j>`/`pidx_<j>` is produced only by the mask rows, whose text is below `2^j` for any operand; a `wgown_<j>` is consumed only by rows whose text guards it below `2^j`. The array's length is `2^j` by the map, so every index is in range for the array it indexes (GOAL item 4, first bullet). Two arrays never share a dual with different lengths, which was A3 3.2's hole. A read of an input is clamped by its own text. The map's total is within the target by validation, and every array a tree can name is declared by the prologue, so `workgroup_array_not_in_grammar` and `workgroup_memory_over_limit` have no tree to fire on.

**Cost.** A kernel's workgroup arrays are the map's, one per stratum per element type (two arrays of one element type and one length are unspellable; a seed that needs them reads as outside the kingdom by reason and S1 reports it, so a problem whose seeds need more declares a map with the strata they use); lengths are powers of two. A read past a read-only binding's end returns its last element, by the row's text: free `phi[t*d + c]` addressing is kept. The buddy-tree of gene-carried declarations is withdrawn: a declaration subtree cannot bind uses elsewhere in a gene, so a tree could use an undeclared array or declare a set over the limit.

**Removed.** `u32.wgindex/wgvindex/wgv2index/wgaindex/pindex/pindex16/pindex128` (replaced by the class duals), `safe.wg_of/wgv_of/wgv2_of/wga_of/pidx64/pidx16/pidx128`, `u32.wglen`, `literal.wglen`, `wg.decl*`, `wg.decls*`, `grid.with_wg`, `wg.decl` and `wg.decls` duals, the raw `wg.get`/`wg.set`, `load.buffer` (F007), `u32.phi`, `load.phi_safe` (now `load.<input>`), `workgroup_plan`, `array_len` lookups in the renderer, the renderer's re-mask (S7), `own.u32_wgindex` and friends (now `own.u32_wgidx_<j>` = `0u`, `own.u32_wgown_<j>` = `lidx`), the device bounds policy as the meaning of a read.

### 9.2.3 Write ownership and disjointness per output binding (layer 7; layer 3 `store_cell_wraps`)

Reasons: `write_sets_overlap`, `store_index_not_affine`, `store_index_may_wrap`, `store_stride_may_be_zero`, `index_not_in_bounds` (writes), `store_cell_wraps`, `output_read_of_another_invocations_cell`; the mirror (`u32.mirror`, `diag_twice`); A3 2.6's mixed block-form pairs; renderer S4 (fallback own leaves), S8 (offset clamps).

**Types.** For each written binding `b` of the entry that is not an arena (the alias class stands for its members): `u32.own_<b>` — "a cell of `b` this invocation owns, for reading". The invocation's origins: `u32.rowbase`, `u32.colbase` (the first row and column of its block, under the contract's partition), and `u32.threadrow`, `u32.threadcol` (the origin plus a clamped offset). The six partition-specific origin duals collapse into these two, because the partition is a contract fact spelled by the rows, not a choice made per node. A store takes a row and a column, never a flattened cell (the store row must spell its own guard from its own inputs, §9.4 rule 1).

**Rows.** Origins: `safe.thread_row: vec3<u32>.gid -> u32.rowbase` and `safe.thread_col: vec3<u32>.gid -> u32.colbase`, spelled by the partition: contiguous `gid.y * KNOB_BROW`, strided `wid.y * (KNOB_WROW * KNOB_BROW) + lid.y`, grouped `wid.y * (KNOB_WROW * KNOB_BROW) + lid.y * (KNOB_BROW / G)`, **lane** `wid.x` for the row and `lidx` for the column (block 1 on both axes). The knobs are the kernel-scope constants of §9.2.10. Offsets, each taking an origin and clamped into the block, as today: `safe.row_plus: (u32.rowbase, u32.lit64) -> u32.threadrow` spelled `o + min(i, KNOB_BROW - 1u)` (contiguous), `o + min(i, KNOB_BROW - 1u) * KNOB_WROW` (strided), `o + min(g, G - 1u) * (KNOB_WROW * KNOB_BROW / G) + min(e, KNOB_BROW / G - 1u)` (grouped, `safe.row_plus_g` with two offsets), `o` (lane: the block is one cell, so the offset rows are absent from a lane kingdom); the tile-counter and any-expression offsets `safe.row_plus_tile<d>`, `safe.row_plus_x` likewise; `safe.row_as_index: u32.threadrow -> u32.index` and `safe.rowbase_as_row: u32.rowbase -> u32.threadrow` (the origin as a row, offset zero). Stores take the row and the column: for a 2-D output `b` with the contract's row stride `S_b` (a param such as `cfg.d`, or the lane width `W`), `store.<b>_<rung>: (u32.threadrow, u32.threadcol, <elem>) -> stmt.<rung>` spelled `if (G_b) { b[r * S_b + c] = v; }`; for a 1-D output `store.<b>_<rung>: (u32.threadcol, <elem>)` spelled `if (G_b) { b[c] = v; }`. `G_b` is the row's own text over its own inputs `r`, `c`: the generator proves which guard is sufficient (§9.6.2 item 1): none when the template is in bounds for every value of the params, else `r < R_b && c < C_b` from the length polynomial (the Gram's `row < r_rows && col < d`), and for a lane partition `lidx < Wu` (an invocation past the row's width owns no cell and writes none). Accumulating stores `store.<b>_add_<rung>` spell `b[r * S_b + c] = b[r * S_b + c] + v` under the same guard. Cells for reading: `safe.cell_<b>: (u32.threadrow, u32.threadcol) -> u32.own_<b>` spelled `(r * S_b + c)` (1-D: `c`); `load.<b>_own: u32.own_<b> -> <elem>` spelled `b[min(i, arrayLength(&b) - 1u)]`, the only load row of a written binding (the clamp is the row's text, as for inputs; the generator's proof makes it the identity on owned cells). Own terminals: `own.u32_rowbase`, `own.u32_colbase` (the origins themselves), `own.u32_threadrow`, `own.u32_threadcol` (origin, offset 0), `own.u32_own_<b>` (the invocation's first cell).

The mirror: in `K<..., mode=full>` only, `store.<b>_mirror_<rung>: (u32.threadrow, u32.threadcol, <elem>) -> stmt.<rung>` spelled `if (r < c && G_b) { b[c * S_b + r] = v; }`: the triangle guard is the row's text over its own inputs, so the diagonal is written by the own store only; no mirror value dual exists.

**Closure.** A store of `b` is a row whose index inputs are `u32.threadrow` and `u32.threadcol`, which are produced only by the origin and offset rows, which read the invocation's coordinates and the grid's knobs. So every store of `b` in every tree writes `origin(invocation) + clamped offset`, with one partition and one block factor for the whole kernel (both are grid and contract facts spelled into every origin row, so two store sites of one binding in one kernel share them). The generator proves, per contract, once, that the set of cells `{origin(t) + o : 0 ≤ o < block}` is disjoint across invocations `t` of the dispatch (for `lane`: `wid.x * W + lidx` under `lidx < W` is injective over the workgroups × lanes) and that `r * S_b + c` does not wrap over the param ranges, by interval arithmetic over the contract and by the exhaustive enumeration of §9.6.2 at the contract's smallest shapes; a template that fails is not emitted and the instantiation report names it. Hence `write_sets_overlap`, `store_index_not_affine`, `store_stride_may_be_zero`, `store_index_may_wrap` and `store_cell_wraps` have no tree to fire on, and `index_not_in_bounds` is met by `G_b` in the row's text. The only load of `b` takes `u32.own_<b>`, produced only by `safe.cell_<b>` over the same row and column duals, so no invocation reads a cell another writes. The mixed-pair hole of A3 2.6 closes because the partition is one per kernel.

**Cost.** Stores to a non-arena output are at `invocation origin + bounded offset` under one partition per kernel (contiguous, strided, grouped or lane); scatter and data-dependent stores go to arenas (§9.2.4) or through atomics; a transposed write is the mirror template under `mode full` only; two partitions in one kernel are unspellable; under `lane W` a workgroup writes one row of `W` cells, one cell per lane, and a design whose invocations share a row's cells (today's rows mode) is outside the kingdom by reason, which S1 reports for each V24 `cands` writer. Exactly the price layer 7's Revision 1 already charges (non-affine stores illegal). The per-entry `writes` and `partition` facts are declared by the problem (`problem.toml`, lead's decision 4); the instantiation refuses an entry whose written binding has no partition fact.

**Removed.** `u32.cell`, the nine `safe.cell_of` pairs, `u32.threadrows/threadcols/threadrowg/threadcolg`, `u32.rowbases/colbases/rowbaseg/colbaseg`, `safe.thread_row_s/_g`, `safe.*_s` and `_g` offset rows (one family under the partition fact), `store.h_*` over `u32.cell`, `load.h_safe`, `load.h`, `store.h` (role), `store.buffer` (F195), `u32.blk` as an argument of the origin rows (it is the grid's), `u32.mirror`, `safe.mirror_of` and the mirror rows outside `mode full`, the `rows` policy and `u32.ownrow_<r>` (now `partition lane`), `render::own_fallback`, `fallback_value`, `own_leaf` as a substitute (S4), `Contract::for_role`'s layer 7 on the evolution path, the renderer's offset clamps as a separate step (S8: the clamp is the row's text).

### 9.2.4 Arenas: the policy as a contract choice

Reasons: `arena_index_not_in_bounds`, the `--layer7 strict | arena | arena+rows` policy (A1 P6), `index_not_in_bounds` on arena bindings. (Today's `rows` mode is the `lane` partition of §9.2.3.)

**Types.** For each arena binding `a` (`policy arena a`): `u32.aidx_<a>` — "an index below `arrayLength(a)`, by the mask or modulus its row writes"; the contract's `param arrayLength(a) L L` fixes `L`.

**Rows.** `arena.idx_<a>: u32.index -> u32.aidx_<a>` spelled `(i & (L - 1)u)` when `L` is a power of two (V24: `8,388,608 = 2^23`), else `(i % Lu)`; `load.<a>: u32.aidx_<a> -> <elem>` spelled `a[i]` for a plain array; `store.<a>_<rung>: (u32.aidx_<a>, <elem>) -> stmt.<rung>`; for an atomic array the loads and read-modify-writes of §9.2.12 (`atomic.<a>_load: u32.aidx_<a> -> u32.atomicold`, `atomic.<a>_<fun>: (u32.aidx_<a>, <elem>) -> <atomicold>`).

**Closure.** Every arena access is at an index below `L` by its row's text. Plain stores to an arena at colliding indices give WGSL's indeterminate value, not a fault; the contract declares arenas as the design's own state, and §9.2.12 types what an in-kernel read of an arena may feed.

**Cost.** The arena policy is a kingdom, not a flag: a kernel of `K<policy strict>` cannot touch an arena.

**Removed.** `Contract.arena_mode`, `rows_mode`, `Contract.rows`, `--layer7`, `kernel_info`'s mode handling of these; the role's `store.<name>_<rung>` and `load.<name>` at `u32.index` for atomic arrays (now at `u32.aidx_<a>`).

### 9.2.5 Workgroup memory races and ordering: phases (layer 7)

Reasons: `workgroup_memory_race`, `memory_barrier_without_control`; renderer S10 (statement erasure), `wg.put`'s double barrier.

**Types.** Two new block kinds at every depth `d`: `block.W<d>` / `stmt.W<d>` — "a write phase: statements between a leading and a middle barrier; writes to workgroup arrays at this thread's own elements only, reads of workgroup arrays at this thread's own elements only, no barrier"; `block.WB<d>` — a write-phase loop body that breaks. The existing `L`, `C`, `O`, `B` kinds keep their meanings and gain: "reads workgroup arrays at any index, writes none".

**Rows.** The phase rows, at the barrier rungs only: `wg.phase_L<d>: (block.W<d>, block.L<d>) -> stmt.L<d>` spelled `workgroupBarrier(); { W } workgroupBarrier(); { L }`, and `wg.phase_s_L<d>` whose separators are `storageBarrier(); workgroupBarrier();` (the storage barrier is thus always inside a control barrier row). `wg.put_<e>_<j>_L<d>: (wgmem<e,2^j>, u32.wgown_<j>, e) -> stmt.L<d>` keeps its spelling `workgroupBarrier(); if (i < 2^j u) { a[i] = v; } workgroupBarrier();` (a one-write phase). `stmt.barrier_L<d>` is kept as the phase with an empty write block, spelled `workgroupBarrier();` (lead's decision 6). Writes: `wg.set_<e>_<j>_W<d>: (wgmem<e,2^j>, u32.wgown_<j>, e) -> stmt.W<d>` spelled `if (i < 2^j u) { a[i] = v; }` and the lane form `if (i < 2^j u) { a[i][(l) & 3u] = v; }`; the only writers of workgroup arrays, and they output `stmt.W<d>` only. Own reads in W: `wg.get_own_<e>_<j>: (wgmem<e,2^j>, u32.wgown_<j>) -> e` spelled as in §9.2.2 (guarded, never clamped onto another invocation's element), the only workgroup read row a W statement can reach. Foreign reads: `wg.get_<e>_<j>` over `u32.wgidx_<j>` anywhere outside W. Reductions `wg.reduce_*_L<d>`, `wg.uload_*_L<d>` keep their spellings (each begins with a barrier). Workgroup atomics: the rows of §9.2.12, in the kinds that section names. Loops in W bodies take `block.W<d+1>` / `block.WB<d+1>` bodies; `if` in W takes `block.W<d>` bodies.

Value typing inside W (lead's decision 5: both forms built in B1, their widths measured, the narrower one that spells every seed kept). Values are rung-free in this design (a `f32.real` is one dual everywhere), so a W statement must be kept from reaching a foreign read through any value. Form A: the W kind's statement rows take their values from a second value family (`f32.realw`, `u32.indexw`, ...) whose producers are the value rows over that family and the own reads, never `wg.get_<e>_<j>`. Form B: the value family is shared and the W kind's statement list is restricted (register sets, private-array sets, own writes, atomics) while the foreign-read rows output a third family of duals (`f32.realr`, ...) that only L/C/O/B statement rows consume. Either form is sound only if no value dual a W statement can demand reaches, through any chain of producers, a row that reads a foreign element: §9.6.2 item 7 computes the reachability closure over the generated table and asserts it (the advisor's finding 9), for whichever form is built.

**Closure.** The spelling `barrier; W; barrier; L` orders every write (in W) after every read of the preceding L code and before every read of the following L code. Within W, writes are at `u32.wgown_<j>` under the guard `i < 2^j`: `i = lidx + k · THREADS` with `k` a literal or a uniform tile counter, so for one `k` the map `lidx ↦ i` is injective over the workgroup's invocations and no two invocations of one workgroup write one element (the clamp of revision 1 broke this: at `THREADS = 256`, `j = 8`, `k = 1` every lane clamped to 255; §9.6.2 item 7 now asserts injectivity for every `k` of every row's domain). The only reads in W are own reads at the same guarded index, so no element is both read by one invocation and written by another between two barriers. The reductions, puts and uniform loads carry their own barriers in uniform control (L rungs). Barriers exist only in rows outputting `stmt.L<d>`, so a memory barrier always sits in a control barrier row. Hence no tree has a workgroup race.

**Cost.** A read-modify-write of a foreign element within one phase is unspellable; an unsynchronised shared pattern (one invocation writes, another reads without a barrier) is unspellable, as intended. Two more block kinds per depth, and one extra value family (form A or B).

**Removed.** `wg.set_<r>` over `u32.wgindex` (any masked index) at every rung, `wg.set_v_lane_<r>` likewise, `stmt.storage_barrier_<r>` as a free statement (it is `wg.phase_s_L<d>`), the clamp in `safe.wg_own*`, `render::scope.masked` and the erasure branches (S10), `Rendered.masked`, `is_barrier` in the oracle's grammar layer.

### 9.2.6 Statement grammar, rungs, barriers and uniformity (layers 2, 4; layer 7 uniformity rules)

Reasons: `barrier_under_condition`, `break_outside_loop`, `continue_outside_loop`, `block_not_in_safe_grammar`, `statement_not_in_safe_grammar`, `loop_not_bounded_constructor`, `nested_item_loop`, `tile_depth_exceeded`, `barrier_in_nonuniform_control_flow`, `barrier_in_nonuniform_control`, `subgroup_operation_in_nonuniform_control`, `loop_bound_not_uniform`; renderer R2.

**Types.** As today: `stmt.L<d>`, `block.L<d>` (`d` in `0..=5`), `stmt.C<d>`, `block.C<d>`, `stmt.O<d>`, `block.O<d>` (`d` in `0..=4`), `block.B<d>` (`1..=5`), plus `W<d>`, `WB<d>` of §9.2.5, each at the fuel strata of §9.2.8 (`_f<k>`). `u32.uniform`, `bool.uflag`, `u32.boundu` as today. Loop counters split by uniformity: `u32.counter` (the item loop with a uniform bound) and `u32.xcounter0` (the item loop with an expression bound); `u32.tcounter<d>` (a tile loop at an L or O rung whose bound is a literal or `u32.boundu`: uniform) and `u32.xcounter<d>` (a tile loop at a C, B or W rung, or any loop that breaks: divergent).

**Rows.** As today for blocks, `if`, `if_else`, `cblock`, `return_L0..2`, `cont_*`, `brk_B<d>`, loops at every rung, the `if_uflag`, `if_else_uflag`, `if_u{le,eq,ne,lt}` uniform tests, `uflag.*`, `uniform.*` (post-A0: no `of_flag`, no `of_checked`); loop rows at L and O rungs take `u32.lit64`/`u32.lit4096`/`u32.boundu`/`u32.boundm` and bind `u32.tcounter<d>`; loop rows at C, B, W rungs and every `brk` loop take any bound (`u32.boundx<c>`, §9.2.8) and bind `u32.xcounter<d>`; `uniform.of_counter: u32.counter -> u32.uniform`, `uniform.of_tcounter<d>: u32.tcounter<d> -> u32.uniform` (no row takes an `xcounter` into `uniform`). Counter reads: `safe.tile_counter<d>: -> u32.tcounter<d>` and `safe.xtile_counter<d>: -> u32.xcounter<d>`, each spelled as its variable (`k<d>`, `kx<d>`, `t`), declared once at kernel scope and initialised `0u`, and never redeclared: a loop spells `k<d> = 0u; for (; k<d> < n<d>; k<d> = k<d> + s) { ... }` (§9.2.7), not `for (var k<d> = ...)`, which would shadow the kernel-scope variable and make every read after the loop `0u`. So before its loop a counter reads `0u`, inside it the trip, after it the final value; after an unrolled loop (`unroll_*`, whose copies carry the counter as a literal and never assign the variable) it reads what it held before, `0u` if no earlier loop of that depth ran. Today's `lit_u32(0u)` rewrite for a read outside the loop (S5) is gone; the reader reads a kernel's counter use after its loop as the final value, exactly (lead's decision 7).

Subgroup operations become statement rows at the L rungs: `subgroup.<op>_<reg>_L<d>: (reg<T>, T) -> stmt.L<d>` spelled `r = subgroupOp(v);` for every collective and gather (`add`, `mul`, `min`, `max`, inclusive and exclusive scans, `and`, `or`, `xor`, `all`, `any`, `broadcast`, `broadcast_first`, `shuffle*`, `quad_*`, `ballot`), one per register kind the result has; `stmt.subgroup_barrier_L<d>` as today. The subgroup builtins (`subgroup_invocation_id`, `subgroup_size`, `num_subgroups`, `subgroup_id`) stay value terminals.

**Closure.** The rung structure is unchanged from A3's YES rows: a `C`, `B`, `W` block holds no row that outputs a barrier-bearing statement; every `if` over a `bool.flag` drops to a `C` body; every `if` that keeps the rung takes `bool.uflag` or two `u32.uniform`, whose producers are the contract's uniform fields, the workgroup id, the grid, literals, arithmetic of uniforms, and uniform counters; loops that keep an L body take uniform bounds; loops whose body is `C`/`B`/`W` bind `xcounter`, which no row makes uniform. A barrier (phase, put, reduce, uload, subgroup barrier, subgroup statement) is a row outputting `stmt.L<d>` only, so it is reached only through L blocks, whose every enclosing condition and bound is uniform. A conditional return has no row. Nesting (`nested_item_loop`, `tile_depth_exceeded`) is the rung ladder: no row takes `block.L6`. Subgroup operations are statements at L rungs, so they are in uniform control.

**Cost.** A subgroup result passes through a register (one statement) instead of sitting in an expression; a counter of a divergent loop is never a uniform value (it was never one).

**Removed.** `uflag.of_flag`, `uniform.of_checked` (A0), `subgroup.*` value rows (71 templates), `uniform.of_tcounter<d>` over a C-rung loop's counter (split), the oracle's grammar layer on the path, R2's renderer branches.

### 9.2.7 Bounded loops and the body-invariant bound (layer 6)

Reasons: `loop_bound_wraps`, `loop_bound_changes_in_body`; renderer S6 (loop caps).

**Types.** Trip classes: `u32.lit64` (`1..=64`), `u32.lit4096` (`1..=4096`), `u32.boundm`/`u32.boundd`/`u32.boundrows` (each present only when the contract has the param, carrying the param's range), `u32.boundu` (uniform, capped), `u32.boundx6`, `u32.boundx12`, `u32.boundx18`, `u32.boundx24` (an expression bound capped at `2^c`), `u32.xbound` as today (the value of an expression bound).

**Rows.** `bound.of_index_<c>: u32.xbound -> u32.boundx<c>` spelled `min(e, 2^c u)`; `bound.of_uniform_<c>: u32.uniform -> u32.boundu<c>` likewise; `bound.m: -> u32.boundm` spelled `cfg.m`. Every loop row spells its bound into a `let` before the loop and resets the kernel-scope counter without redeclaring it: `let n<d>_<id> = <bound>; k<d> = 0u; for (; k<d> < n<d>_<id>; k<d> = k<d> + s) { body }` (`id` a per-loop suffix the renderer numbers in order, so a second loop at the same depth gets its own `let`; `items_cont_*` omits the reset), so the trip count is fixed at entry whatever the body writes. Strides `s` are literals in `1..=4096`.

**Closure.** The bound is read once, before the body, into an immutable `let`, so `loop_bound_changes_in_body` has no tree to fire on. The counter steps by `s ≤ 4096` to a bound of at most `2^24` (expression and uniform bounds) or at most the contract's param range (a binding bound, whose range the generator checks is at most `2^32 - 4096`, else the bound row is capped like an expression bound), so the counter never steps past `u32::MAX`: `loop_bound_wraps` has no tree. The `xbound` rows cannot grow past their operands except `index.add`/`max`/`or` of two bounds, which may wrap to a *smaller* value: a smaller bound is still a bound, and the cap row makes it finite.

**Cost.** Trip counts above `2^24` are unspellable; a loop's bound is evaluated once (a loop whose bound a seed changes inside its body reads as outside the kingdom by reason, C4).

**Removed.** `MAX_LOOP_TRIP` as a renderer constant (the cap is the row's text), the single `u32.boundx`, `bound.of_index`, `bound.x_of_uniform` (uncapped) and `wg.get_u_safe -> u32.xbound` (a workgroup read as a bound value stays, through `u32.xbound`), the oracle's layer 6 on the path.

### 9.2.8 The execution budget: fuel strata (the Metal hang, the deadline, predicted time, the runaway drain)

Reasons: A1 P3 (exit code 3), H6 (`slow: predicted`), H10 (`runaway`), the cost estimator (`render::cost`), the harness deadline.

**Types.** Every statement-level dual carries a fuel stratum `k` in `0..=4`: `stmt.L2_f1`, `block.C3_f2`, `block.W1_f0`, ... Stratum `k` means: executing this statement once performs at most `N · 2^(6k)` node executions, where `N` is the layout's program size (every expressed function executes at most once per iteration of its enclosing loops, and a loop-free subtree has at most `N` functions). Trip classes: `c1 = 2^6` (`lit64`, `boundx6`, `boundu6`), `c2 = 2^12` (`lit4096`, `boundx12`, `boundu12`, and a binding bound whose contract range is at most `2^12`), `c3 = 2^18` (`boundx18`, `boundu18`, binding bounds up to `2^18`), `c4 = 2^24` (`boundx24`, `boundu24`, binding bounds up to `2^24`; a binding bound above `2^24` has no row).

**Rows.** A loop row of trip class `c_i` over a body of stratum `k` outputs stratum `k + i`, and exists only when `k + i ≤ 4`: `loop.tile_L0_f2: (u32.lit64, block.L2_f1) -> stmt.L0_f2`, `loop.items_boundm_f3: (u32.boundm, block.L1_f1) -> stmt.L0_f3` (with `cfg.m ≤ 2^12`), and so on for every loop row of §9.2.6 including unrolled loops (an unrolled body is `n` copies: the same class). Sequence rows take statements of one stratum: `block.seq4_L0_f2: (stmt.L0_f2 × 4) -> block.L0_f2`; a lift row raises a statement one stratum without changing its text: `stmt.lift_L0_f2: stmt.L0_f1 -> stmt.L0_f2`. `if`, `if_else`, `cblock`, `block_`, `cont_`, `brk_`, the phase rows and every statement row that is not a loop keep the stratum of their blocks. The kernel root takes the contract's budget: `kernel.entry_safe: (grid.grid, block.L0_f<K>, block.O0_f<K>) -> kernel.safe` with `K` the `budget` line, and own terminals exist at every stratum.

**Closure.** Let `F(s)` be the number of expressed functions in the subtree of a statement-level node `s`; `F(root) ≤ N` by the layout. Claim, by induction on the subtree: a node of stratum `k` executes at most `F(s) · 2^(6k)` node executions per execution of `s`. A loop-free statement executes each of its functions once: `F(s) · 2^0`. A loop of class `c_i` over a body `b` of stratum `k` executes its header's functions once and `b` at most `2^(6i)` times: at most `F(s) + 2^(6i) · F(b) · 2^(6k) ≤ F(s) · 2^(6(k+i))`. A sequence, `if`, `cblock`, `brk`, `cont` or phase node `s` with children `b_1..b_m` of stratum `k` executes its own functions once and each child at most once: `F(s) + Σ F(b_m) · 2^(6k) ≤ F(s) · 2^(6k)` since the children's subtrees are disjoint parts of `s`'s subtree. A lift row changes the stratum only upward. So a kernel of root stratum `K` performs at most `N · 2^(6K)` node executions per invocation. The dispatch performs at most `I · N · 2^(6K)`, `I` the contract's largest invocation count. Every loop is bounded (§9.2.7), so every kernel terminates; this is the proof. The **wall bound** is `I · N · 2^(6K) / E_target`, where `E_target` is the target's execution rate, measured once per target as the slowest node family of the kingdom (a global memory load at stride, a subgroup gather, a transcendental) under the worst cache behaviour the harness can arrange, divided by a safety factor of 10; the contract's validation (9.1.2 item 7) requires the wall bound at the chosen `K` to be within the problem's `base_ms`. The harness's watchdog is then an infrastructure fault only: a dispatch that outlives 10 × the wall bound is a device fault, logged with its kernel, never a score, and the run stops until the fault is understood (a kernel that provably executes at most `I · N · 2^(6K)` nodes and still outlives the bound is evidence against `E_target`, which C1's dispatch stage measures over a million kernels: the maximum observed node rate against `E_target`). **OPEN 8:** `E_target` is a measurement with a margin, not a theorem; Codex's rev4 finding 1 stands exactly to that extent, and the design makes the extent explicit and measured.

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

**Rows.** `literal.<form>: -> <dual>` is the placeholder `?_T` of its dual; a `Dc_T` word names a value of the domain, and Dc mutation, recombination and transposition select valid `C_T` indices (§4). Knob references `literal.p_<knob>: -> u32.index | u32.lit64 | u32.lit4096` carry a `u32.payload`-domain constant `p` and spell, from the node itself, the WGSL expression `clamp((KNOB_<knob> * lit_u32(p)) / 256u, lo, hi)` over the kernel-scope constants `const KNOB_WCOL: u32 = ...;` ... that the prologue emits from the grid node (`wcol, wrow, bcol, brow, colspan, rowspan, threads, ktile`; `ktile` is the fourth argument of `grid.of`, default `1`, so every knob exists in every kernel); `lo..=hi` is the output dual's domain. The row's text depends on its own constant only; the reader reads the expression back to the row and its payload exactly. `_log2` and `_m1` forms spell `firstLeadingBit(...)` and `... - 1u` over the same expression.

**Closure.** A literal's value is a member of its domain by construction of `C_T`; the walk expresses `?_T` only as a leaf; every knob a reference names is a constant of every kernel's prologue, so `knob_reference_unresolved` has no tree. `literal_with_children` cannot occur because a literal row has no signature rows.

**Cost.** None.

**Removed.** `render::snap`, `lit` clamping (S3), `params::resolve` (a value computed from another node at render time), `literal.p_*` over `u32.wglen` (no wglen).

### 9.2.11 Render and parse (layer 7 `kernel_does_not_render`, `kernel_does_not_parse`; R1–R4)

**Types and rows.** None new. The renderer is a total function `render: Tree -> String` (§9.4): every row of every generated table has a spelling; the generator's test renders every row once at an instance of each input (§9.6.2 item 3) and parses the text in naga with the target's capabilities. The `load.cfg_*` rows spell the interface's own field names (A0 fixed `pcfg.d`).

**Closure.** Every node is a row with a spelling (frame membership), so `render` has no `Err` branch to take; `kernel_does_not_render` and `kernel_does_not_parse` are generator defects caught by §9.6.2 item 3, not properties of a gene. R2 is the rung ladder (§9.2.6), R3 the domains and the target (§9.2.1, §9.2.10), R4 the per-entry alphabet (an interface's rows are the only rows; a workgroup array is consumed only by access rows; every helper has a scalar by the generator's test).

**Removed.** Every `Err` return of `render.rs` on an expressed tree (A1 R1–R4), `Rendered.masked`, `Checker::safety`'s render-and-parse on the path.

### 9.2.12 Atomics and determinism

Reasons: A3 7.6 (determinism), Codex rev4 finding 2, `statement_not_in_kingdom` (ray queries, image atomics, cooperative stores: no rows).

**Types.** `u32.atomicold`, `i32.atomicold` — "an order-dependent atomic value: the old value a read-modify-write returned, or an in-kernel load of an arena atomic that other workgroups may be updating". No row of a strict kingdom converts it to a plain value.

**Rows.** Read-modify-writes output the `atomicold` duals and are statements' values only where they are race-free: workgroup atomics `atomic.add_u_<j>: (wgmem<atomic<u32>,2^j>.index, u32.wgidx_<j>, u32.index) -> u32.atomicold` (and `sub`, `min`, `max`, `and`, `or`, `xor`, each `_u`/`_i`); arena atomics `atomic.<a>_add: (u32.aidx_<a>, u32.index) -> u32.atomicold`. Their consumers in `determinism strict` are `stmt.discard_u_<rung>: u32.atomicold -> stmt.<rung>` (`_ = e;`) and nothing else. Loads: a workgroup atomic's value is read by the statement row `atomic.uload_u_<j>_L<d>: (reg<u32>.index, wgmem<atomic<u32>,2^j>.index, u32.wgidx_<j>) -> stmt.L<d>` spelled `workgroupBarrier(); r = atomicLoad(&a[i]); workgroupBarrier();` (the `wg.uload` form): the barriers order it after every read-modify-write of the preceding code and before any following one, and `atomicLoad` after a barrier returns the final value of a commutative reduction, the same in every invocation and under every order. An arena atomic's in-kernel load `atomic.<a>_load: u32.aidx_<a> -> u32.atomicold` outputs `atomicold` (no barrier spans workgroups; the harness reads the arena's final value after the dispatch). `atomic.exchange_*` and `atomicCompareExchangeWeak` are absent from strict kingdoms. In `determinism relaxed` (the V24 contracts, whose designs allocate slots by `atomicAdd`): the conversion rows `index.of_atomicold: u32.atomicold -> u32.index`, `int.of_atomicold`, the plain-valued arena load `load.<a>: u32.aidx_<a> -> u32.index` spelled `atomicLoad(&a[i])`, and `exchange`. The harness's output kind states what fitness compares (V24: pooled recall, order-free).

**Closure.** In a strict kingdom every atomic read-modify-write's result is `atomicold`, whose only consumer is `discard`; the only workgroup atomic load is barrier-bracketed into a register, so its value is the reduction's final value; the only arena atomic load is `atomicold`. So no row carries an order-dependent value into a register, an index, a store or a bound, and every stored value is a deterministic function of the inputs (Codex rev4 finding 2, the advisor's finding 2). Race-freedom: an atomic access races with nothing; the uload row stands at L rungs, in uniform control. In a relaxed kingdom the dependence is declared by the contract and the fitness measure is order-free by the problem's declaration.

**Cost.** Strict kingdoms spell histograms and reductions through atomics with their results read after a barrier, one register statement per read; slot allocation needs a relaxed contract.

**Removed.** `atomic.*_u`/`_i` outputting `u32.index`/`i32.int` directly; `atomic.load_u`/`load_i` as value rows (`gen_tables.py:995`); the role's plain `load.<a>` of an atomic arena from strict kingdoms; `atomic.exchange` from strict kingdoms.

### 9.2.13 The typed tree and the root (layers 1, 2 root; `module_entries_not_the_interfaces`; `no_own_terminal`)

Reasons: `unknown_dual`, `unknown_row`, `signature_mismatch`, `no_own_terminal`, `root_not_safe_kernel`, `module_entries_not_the_interfaces`, `grid_not_in_grammar`.

**Types and rows.** The root of a gene of `K_e` is `kernel.safe`, produced only by `kernel.entry_safe: (grid.grid, block.L0_f<K>, block.O0_f<K>) -> kernel.safe` when the contract names a `primary` output (the epilogue's last statement is a store to `u32.own_<primary>`), else `kernel.entry_safe: (grid.grid, block.L0_f<K>) -> kernel.safe`. A module is the chromosome's fixed linker (§9.1.4). Every demanded dual has a terminal (`docs/PLAN_wgsl_terminals.md`, delivered 034bf41f); the `own.*` rows are real terminals with their own text. The tables' fallback columns (`fallback`, `fallback_bits`, `fallback_noop`, `fallback_elem`) are removed (A0).

**Closure.** The kingdom check of §1 and the walk of §3: every node is a row at its signature, every tail cell a terminal of its type, the root a producer of the root type. These reasons are the kingdom check's, run once per table, never per gene.

**Cost.** None. An entry's epilogue store is a problem fact (`primary`, lead's decision 10): for an entry with no `primary` output (V24 `ingest` writes arenas only) the "every gene writes its output" purpose of the epilogue is carried by fitness.

## 9.3 Layout bounds for one V24 kingdom at `N = 8`

The bounds are those of §2 and `KingdomContract::bounds`; the build measures them in B1 by the kingdom check (`the_kingdom_reports_print`); this section gives the derivation and an estimate so the design is seen to be buildable.

**Type count of `K<M3 Max, v24_ingest5b, search, contract, strict aliases, partition lane 64 on cands, policy arena index_u32 index_f32, determinism relaxed>`.**

| family | duals today | duals in `K` | why |
|---|---|---|---|
| value duals (scalar, vector, matrix, register, array families) | 158 | about 165 | − partition origins (8), − `wgindex`-style (7), − `mirror`, `cell`, `phi`, `wglen`, `decl`, `decls`, `payload` kept; + `own_cands`, `aidx_index_u32`, `aidx_index_f32`, the stratum map's array duals (one per listed `(e, j)`, about 8) with `wgidx_<j>` × 4 strata, `wgown_<j>` × 4, `pidx_<j>` × 3, `xcounter<d>` × 6, `atomicold` × 2, `shift`, `boundx<c>` × 4, `boundu<c>` × 4, `passes`, `shape` |
| W-value family (§9.2.5, form A or B) | 0 | about 60 | the scalar, register and private-array duals a W statement's values range over |
| statement-level duals | 44 | 61 × 5 strata = 305 | L, C (12 each), O (10), B (5), W (12), WB (5), kernel, grid; × fuel |
| **total** | **202** | **about 530** | |

**Arities.** `n_U` is 4 for every statement and block type (`seq4`, `row_plus_g`, `grid.of`), 3 for `u32.index` (`index.clamp`), 2 or 3 for the rest; `n_max = 4` as today. `a_T` for the widest demands: `a_{stmt.L<d>_f<k>} = 4` (`seq4`), `a_{u32.index} = 3`, `a_{f32.real} = 3` (`fma`, `clamp`), `a_{u32.uniform} = 3` (`uniform.clamp`), `a_{block.*} = 2` (`if_else`, phase), `a_{u32.own_cands} = 1`.

**Heads at `N = 8` (rev 4 §2.1, `h_T = [T = R] + 7 · a_T` over the reachable types).** `h_root = 1`; every statement and block dual of stratum ≤ K is reachable from the root through the lift and sequence rows, so `h = 28` for each `stmt.*` dual and `h = 14` for each `block.*` dual; `h = 21` for `u32.index`, `f32.real`, `u32.uniform`; `h = 7` for a dual demanded once per function (`u32.own_cands`, `bool.uflag`, the origins); `h = 0` for a dual no function demands (`grid.shape` is demanded by `grid.of`: `h = 7`).

**Width.** Today's measured table (rev 4, 2026-10-20): 202 types, `L = 3,958`, width 58,090 words, the widest type `h 21, P 1,379, t 1,358, d 1,379`. The width is a sum over types of `h_T + t_T + d_T` with `t_T ≈ P_T − h_T` and `P_T = Σ_U h_U · a_{T,U}`, so it scales with the number of reachable producer types times their head lengths. With about 530 reachable types, of which 305 statement-level at `h = 28` or `14`, the estimate is **about 150,000 words a gene at `N = 8`** (2.5 × today's), and about 650,000 at `N = 32`. A million genes at `N = 8` is then about 600 GB through the paged geneframe for C1, so C1 streams in pages as rev 4 says; the levers, if the measured figure must come down, are the fuel strata (5 → 3 costs one factor of trip-class resolution) and the W-value family's form. B1 prints the measured bound table at `N = 8` and `N = 32`, and C1's size is set from it (lead's decision 11).

**Terminals per demanded type.** Every value dual has a `?_T` or an `own.*`; `grid.shape` has `?_shape`; every statement-level dual at every stratum has `own.stmt_*` (an empty statement) and `own.block_*` (an empty block, or for O rungs a store of the thread's first cell of the primary output); `u32.own_cands` has `own.u32_own_cands`.

## 9.4 The renderer as a pure function of the tree

`render(tree, C) -> String` is total on expressed trees. It spells: the prologue from `C` (directives, structs, bindings, the stratum map's array declarations, the helper library, the register and counter declarations), the `// grid`, `// mode`, `// passes` lines and the knob constants `const KNOB_WCOL: u32 = ...;` ... from the grid node and `C`, then the tree, post-order, each node by its row's spelling template with `{0}`, `{1}`, ... for its children. No node is looked up for a reason to do otherwise. The spelling of each family, with every mask, clamp and guard as the row's own declared meaning:

| family | spelling (the row's meaning) |
|---|---|
| `literal.*`, `const.*` | the `C_T` value through the identity helper of its scalar: `lit_u32(5u)`, `lit_f32(0.5f)`, `lit_i32(-3i)`, `lit_bool(true)`; vectors and matrices splat the one value |
| `literal.p_<knob>` | `clamp((KNOB_<knob> * lit_u32(p)) / 256u, lo, hi)` over the prologue's knob constants, from the node's own payload `p` |
| `own.*` of a value dual | the declared value: `lit_u32(0u)`, `false`, `lidx` (`own.u32_wgown_<j>`), the invocation's origin (`gid.x * KNOB_BCOL`), the invocation's first cell (`origin_row * S_b + origin_col`) |
| `own.stmt_*`, `own.block_*` | `{}`; for `O<d>`: `{ store_primary(own cell, lit(0)); }` |
| `builtin.*`, `axis.*` | the builtin's name and component: `gid.x`, `lidx` |
| `arith.*`, `trig.*`, `geom.*`, `int.*`, `bits.*`, `compare.*`, `logic.*`, `select.*`, `shape.*`, `convert.*`, `quantise.*`, `hash.*`, `fixed.*` | the naga operator or builtin, or the composite helper's call, over the children; `index.shl`: `({0} << (({1}) & 31u))`; `index.div`: `({0} / {1})`; `index.div_nz`: `({0} / max({1}, 1u))`; `int.div`: `int_div({0}, {1})` (§9.2.9) |
| `safe.wg_of_<j>`, `safe.pidx_<j>` | `(({0}) & (2^j − 1)u)` |
| `safe.wg_own_<j>`, `safe.wg_own_tile_<j>` | `(lidx + ({0}) * KNOB_THREADS)` (unclamped; the access rows guard it) |
| `safe.thread_row`, `safe.thread_col` | the origin under the contract's partition (§9.2.3): contiguous `gid.y * KNOB_BROW`, strided, grouped, lane (`wid.x` / `lidx`) |
| `safe.row_plus*`, `safe.col_plus*` | `({0} + min({1}, KNOB_BROW − 1u))` and the strided and grouped forms |
| `safe.cell_<b>` | `({0} * S_b + {1})`, or `{0}` for a 1-D output |
| `arena.idx_<a>` | `(({0}) & (L − 1)u)` or `(({0}) % Lu)` |
| `load.<b>` (input) | `b[min({0}, arrayLength(&b) − 1u)]`; `load.<b>_own`: `b[min({0}, arrayLength(&b) − 1u)]`; `load.<a>` (plain arena): `a[{0}]`; relaxed only: `load.<a>` of an atomic arena `atomicLoad(&a[{0}])` |
| `store.<b>_<rung>` | `if (G_b) { b[{0} * S_b + {1}] = {2}; }` with `G_b` over `{0}`, `{1}` the row's own guard (`r < R_b && c < C_b`, `lidx < Wu` for lane, empty when none is needed); `store.<b>_add_*`: the same with `b[i] = b[i] + {2}`; `store.<b>_mirror_*`: `if ({0} < {1} && G_b) { b[{1} * S_b + {0}] = {2}; }` |
| `wg.get_<e>_<j>` | `a[{1}]`; lane form `a[{1}][({2}) & 3u]` |
| `wg.get_own_<e>_<j>` | `select(ZERO_e, a[min({1}, (2^j − 1)u)], {1} < 2^j u)` |
| `wg.set_<e>_<j>_W<d>` | `if ({1} < 2^j u) { a[{1}] = {2}; }`; lane form `if ({1} < 2^j u) { a[{1}][({2}) & 3u] = {3}; }` |
| `wg.phase_L<d>`, `wg.phase_s_L<d>`, `stmt.barrier_L<d>` | `workgroupBarrier(); {0} workgroupBarrier(); {1}` / `storageBarrier(); workgroupBarrier(); {0} storageBarrier(); workgroupBarrier(); {1}` / `workgroupBarrier();` |
| `wg.put_*`, `wg.reduce_*`, `wg.uload_*` | as today's text (barriers inside; the put's guard `if (i < 2^j u)`) |
| `atomic.*` read-modify-write | `atomicOp(&a[{1}], {2})` (an `atomicold` value); `stmt.discard_*`: `_ = {0};`; `atomic.uload_u_<j>_L<d>`: `workgroupBarrier(); {0} = atomicLoad(&{1}[{2}]); workgroupBarrier();`; `atomic.<a>_load`: `atomicLoad(&a[{0}])` (an `atomicold` value) |
| `subgroup.<op>_<reg>_L<d>` | `{0} = subgroupOp({1});`; `stmt.subgroup_barrier_*`: `subgroupBarrier();` |
| `reg.get_*`, `reg.set_*_<rung>`, `parr.*` | the register's name; `r = v;`; `pa[{1}] = {2};` |
| `block.one/seq2/seq3/seq4_*` | the statements in order; `stmt.block_*`: `{ {0} }`; `stmt.lift_*`: `{0}` |
| `stmt.if_*`, `stmt.if_else_*`, `stmt.if_uflag_*`, `stmt.if_u{le,eq,ne,lt}_*` | `if ({0}) { {1} }`, `if ({0}) { {1} } else { {2} }`, `if ({0} <= {1}) { {2} }` |
| `block.brk_B<d>` | `{0} if ({1}) { break; } {2}`; `block.cont_*`: `if ({0}) { continue; } {1}` |
| `loop.*` | `let n<d>_<id> = <bound>; k<d> = 0u; for (; k<d> < n<d>_<id>; k<d> = k<d> + s) { body }` (the counter is the kernel-scope variable, never redeclared); `items_cont_*` without the reset; `unroll_*` the body `n` times with the counter a literal in each copy, the variable untouched |
| `bound.of_index_<c>`, `bound.of_uniform_<c>` | `min({0}, 2^c u)`; `bound.m`: `cfg.m` |
| `safe.tile_counter<d>`, `safe.xtile_counter<d>`, `safe.counter` | the counter variable's name (`k<d>`, `kx<d>`, `t`) |
| `uniform.*`, `uflag.*`, `index.of_uniform` | the same text as the index row or field they stand for |
| `grid.of`, `grid.of_rows_x`, `grid.with_passes` | `@compute @workgroup_size(x, y, 1)` from the shape, the `// grid x y bx by[ rows-x]` line, `// passes N` |
| `kernel.entry_safe` | the entry function: declarations, `{1}`, `{2}` |
| `kernel.module_of_<I>` (the chromosome's linker) | the prologue once, then each gene's entry function in the interface's order |

Rules of the function: (1) the spelling of a row never depends on where the node sits or on any other node (the counter variables and the `KNOB_*` constants are kernel-scope names; a store spells its guard over its own row and column inputs; a knob reference spells its own payload); (2) the generator's test spells every row once and parses the result (§9.6.2 item 3); (3) `render` returns `String`, never `Result`, and records nothing (no `masked`, no events).

## 9.5 Reading (C4) under this section

The reader (`safe_reader.rs`) maps a kernel's text onto the rows of its contract's `K_e` exactly: each text form has one row (`a / b` is `index.div`, `a / max(b, 1u)` is `index.div_nz`, `a << (s & 31u)` is `index.shl`, `a << 3u` is `index.shl_lit`, `b[min(i, arrayLength(&b) - 1u)]` is `load.<b>`). A text that spells something `K_e` has no row for is outside the kingdom by a named reason (a bare `b[i]` read of a storage binding, a store at a non-template index or without its guard, a workgroup write outside a phase, a barrier under a thread condition, a loop whose bound its body changes, a workgroup array not in the stratum map, a shape not in the table, a trip product past the budget, an atomic's old value consumed in a strict kingdom), listed as C4 says. The reader performs no rewrite (`lower_returns_in_loops`, override defaults, `conform`, zero substitution are gone, R1).

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

1. **Ownership enumeration.** For every written non-arena binding, every partition template (contiguous, strided, grouped, lane) and every value of the block factors: enumerate every invocation's owned cells at the contract's smallest dispatch and smallest params, and at the largest params with the smallest dispatch; assert pairwise disjointness across invocations (for lane: injectivity of `wid.x * W + lidx` under `lidx < W`) and that `r * S_b + c` is within `u32` and within the length polynomial under the store row's own `G_b`. (Today's `safety.rs` exhaustive small-domain test, moved to the generator.)
2. **Map budget and shapes.** The stratum map's bytes within `max_workgroup_storage`; one array per listed `(e, j)`; every shape of the table within the limits; `own.grid_grid` legal.
3. **Every row spells and parses.** Each row rendered once with own terminals for its inputs, inside a minimal kernel, parsed and validated by naga under the target; every helper present; every composite has a body.
4. **Feature probes.** One kernel per feature family compiled and dispatched on the target before the family's rows are emitted.
5. **Kingdom check and bounds.** `KingdomRelations::check` passes; the bound table at the layout's `N` is printed to the instantiation report (the measured width of §9.3).
6. **Fuel.** For every loop row, the output stratum equals the body's stratum plus the trip class; no row outputs a stratum above 4; the root takes the contract's `K`.
7. **Phases, transitively, and own-index injectivity.** (a) Compute the reachability closure of the generated table from every dual a `stmt.W<d>`/`block.W<d>`/`block.WB<d>` row demands (through every producer of every demanded dual, to a fixpoint): no row in the closure spells a barrier or a read at `u32.wgidx_<j>` (the advisor's finding 9; both value-family forms of §9.2.5 are tested by this). (b) The only rows spelling a workgroup write output `stmt.W<d>`; the only rows spelling a barrier output `stmt.L<d>`. (c) For every `wgown_<j>` producer and every `k` in its domain (every `lit64` value; every trip of a `tcounter<d>` up to the largest class the row can sit under) and every shape of the table, the map `lidx ↦ lidx + k * THREADS` is injective over `lidx < THREADS`, and every consumer of `u32.wgown_<j>` spells the guard `i < 2^j` (the advisor's finding 1).
8. **Strict atomics.** In a `determinism strict` table, every consumer of `u32.atomicold`/`i32.atomicold` is a `discard` row; every atomic load row of a workgroup array outputs `stmt.L<d>` and spells both barriers; every atomic load of an arena outputs `atomicold`.

### 9.6.3 What C1 does not test

Fitness quality (C5), seed coverage (C4, S1), and `E_target`'s truth beyond the dispatch-stage measurement (OPEN 8).

## 9.7 Decisions on revision 1's open questions

The lead decided the twelve open questions of revision 1 (2026-10-21); the advisor's review found decisions 2, 3, 5, 7 and 12 unsound as worded, and revision 2 replaces them with the forms below. Each is decided toward the goal (every legal kernel spelled; every expressed tree legal; no rewrite).

1. (§9.2.1) Shape table: the target's full legal domain (all legal `(x, y)`), so every legal seed reads (C4). Unchanged.
2. (§9.2.2) Workgroup memory: a fixed **stratum map**, one array per power-of-two stratum per element type, total within `max_workgroup_storage`, declared from the map in the prologue; a use chooses a stratum, which is the evolvable length. The gene-carried buddy-tree of declarations is withdrawn: a declaration subtree cannot bind uses elsewhere in the gene, so a tree could use an undeclared array or declare a set over the limit (`workgroup_array_not_in_grammar`, `workgroup_memory_over_limit` would have a tree to fire on).
3. (§9.2.2) A read of a read-only binding spells its clamp in the row's own text, `b[min(i, arrayLength(&b) - 1u)]`, target-independent; the reader reads a bare `b[i]` as outside the kingdom. Free addressing is kept. The device's bounds policy (naga `Restrict` on Metal, another on Vulkan) is no longer the meaning of any row.
4. (§9.2.3) Per-entry `writes` and `partition` facts: declared by the problem definition (`problem.toml`); B1 adds them for V24 from the problem owner's code. Unchanged.
5. (§9.2.5) W-phase values: both forms (a second value family; a restricted W statement list with a third family for foreign reads) built in B1 and their widths reported; the narrower one that spells every seed is kept, and either is admitted only when §9.6.2 item 7's transitive reachability test passes on its table.
6. (§9.2.5) A bare `workgroupBarrier();` is a phase with an empty write block (`stmt.barrier_L<d>` kept). Unchanged.
7. (§9.2.6, §9.2.7) A loop counter is the kernel-scope variable, never redeclared: a loop spells `k<d> = 0u; for (; k<d> < n<d>_<id>; k<d> += s)`, so a read after the loop is the final value, a read after an unrolled loop is what the variable held (the unrolled copies never assign it, `0u` if no loop of that depth ran); the `0u` rewrite is removed.
8. (§9.2.8) `E_target` is measured per target with a tenfold margin and verified by C1's dispatch stage; a dispatch outliving ten times the wall bound is an infrastructure fault that stops the run, never a score. Unchanged (the extent to which this is a measurement, not a theorem, is stated at OPEN 8).
9. (§9.2.9) i32 division: a probe on naga 30's Metal backend picks the single spelling. Unchanged (OPEN 9 names the probe).
10. (§9.2.13) An entry's epilogue store is a problem fact (`primary`). Unchanged.
11. (§9.3) B1 prints the measured bound table at `N = 8` and `N = 32`; C1's size is set from it. Unchanged.
12. (§9.2.3, §9.2.4) Output ownership is per invocation: `policy rows` is replaced by the partition value `lane W` (origin `wid.x * W + lidx`, block 1, the store row's own guard `lidx < W`), so no two invocations of a workgroup can write one cell by the row's text and the generator's enumeration; §9.2.4 keeps arenas only. S1 reports which V24 `cands` writers fall outside the lane kingdom by reason.

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
| arena index, layer 7 policy | `arena_index_not_in_bounds`, `--layer7` (arena); `rows` mode as `partition lane` | 9.2.4, 9.2.3 |
| `module_entries_not_the_interfaces` | | 9.1.4, 9.2.13 |
| execution budget, device deadline | P3, H6, H10 | 9.2.8 |
| renderer rewrites (ten families) | S1–S10 | 9.2.1 (S1, S2), 9.2.10 (S3), 9.2.3 (S4, S8), 9.2.6 (S5), 9.2.7 (S6), 9.2.2 (S7), 9.2.9 (S9), 9.2.5 (S10) |
| oracle capacity | `too_large_to_prove` | 9.6.1 item 5 |

All 21 ledger rows, the 47 `check.rs` and `safety.rs` reasons of A1 §7 (7 + 13 + 3 + 1 + 3 + 3 + 17 by layer), H1's four facts and H2, the ten renderer families and R1–R4, and the execution budget are assigned.
