# Advisor note: Multi-typed GEP revision 4

Revision 4 of `nucleotable/docs/MultiTypedGEP.md` fixes the five findings of `2026-10-20_multityped_gep_fable.md`. Both SQL queries ran in DuckDB 1.5.6 on the two-type fixture, two genes, two chromosomes, a polymorphic spelling and an ADF reference; the bound query reproduces Karva (`t = d = h(n_max - 1) + 1`) for one type.

## What changed

1. **Bound.** Per producer: `P_T = [T=R] + Σ_U h_U·a_{T,U}`, `L = 1 + Σ_U h_U·(n_U - 1)`, `t_T = max(0, min(P_T, L + h_T) - h_T)`, `Dc_T` length `d_T = min(P_T, L)`. Exact and attained; never larger than revision 3's `H`-bound; one type is Karva unchanged.
2. **Roots belong to the layout.** Root and domain lengths are properties of `(layout_id, gene_id)`; no `gene_root` word on a record. Whole-gene moves (gene transposition, whole-gene parts of one-/two-point, gene recombination) only between homologous gene ids, an enumerated layout set. `g_T` is a layout constant.
3. **ADFs are trees.** Immutable, stored as expressed node rows in the population DAG with `adf_id`, root type and level; a subtree of any size has an encoding. Only genes that vary have a layout.
4. **Specified.** Homeotic head = whole function set + reference terminals; ADF references valid in normal genes; an ADF never references a genic terminal; write-back per type (`j_T <= h_T`), removal-only rewrites always fit, promotion's domain enumerated; `?_T` independent of ADFs; `symbol_id` not spelling in every row.
5. **Head allocation.** `h_T = [T=R] + (N - 1)·a_T` from a declared program size `N`: every program of `N` functions fits whatever its type mix (attained by the chain). Explicit `h_T` kept for the classical SR layout. Width is allocated in full and paged; the kingdom check prints the safewgsl bound table instead of this note estimating it.

## What the built commits must change

### fuller

- `74b03c73` kingdom contract: add `a_{T,U}`, `n_U`, `reachable(R)`; replace `tail_len = min(h·a_T + [T=R], h·(n_max-1)+1)` and `envelope_tail_len` with `P_T`, `L`, `t_T`, `d_T` over per-type `h_T`; `h_T` from `N`; print the bound table per kingdom; `symbol_id` keys. The envelope (a root-independent tail) goes: lengths are per `(layout_id, gene_id)`.
- `0d13d9c1` typed_gep layout and host walk: `GeneLayout::new(k, h, root)` becomes per-type heads from `N` or explicit `h_T`; frames read `head_T` then `tail_T`; `Dc_T` sized `d_T`; add the per-type write-back (`j_T`) as the encoder every rewrite uses.
- `f4f1dde8` tests: bound-exactness constructions (`P_T`, `L + h_T`, the `N` chain), the per-frame locality test, one-type Karva with `h = 1 + (N-1)·n_max`.
- `6c809bd4` DuckDB: replace `nucleotable/sql/typed_gep_expression.sql` with the two rev-4 queries (`symbol_id`, `reference_terminals`, `layouts`/`layout_genes`/`layout_heads`); extend the fixture with the polymorphic spelling and the ADF reference; assert the bound rows.
- `569e27a3` typed_chromosome: `GeneRecord.root` and its `RootMismatch` defect go (root is the layout's); `Definition` bodies become trees (`adf_node` rows), not `GeneRecord`s with a layout; add genic terminals by layout gene id, the homologous-pair enumeration, the homeotic gene over the epoch alphabet, the fixed-linker flag, the no-genic-terminal-in-ADF defect, and ADF references in normal genes.
- `a1eeaa75` keep; add the wrong-output-in-`head_T` defect.
- Rename per §6 in `docs/PLAN_population_dag.md`, `src/population_dag.rs`, `src/homeotic.rs`.
- `docs/PLAN_wgsl_terminals.md` rows are a prerequisite of build step 1 (the live kingdom fails the check today: 129 duals without a terminal).

### phylu

- `65a1cbc7` typed_layout: `KingdomShape` carries `h_T`, `t_T`, `d_T` per `(layout_id, gene_id, type)` from fuller; drop `gene_root` words and buffers, the `envelope_tail_len` sizing and `gene_root_word`; page by `(gene, domain, type, offset)` for heads too.
- `f279b306` walk kernel: root from the layout table, not a `gene_root` word; frames are `head_T` then `tail_T`; Dc cursor bound `d_T`.
- `34bae3a2` typed_init: draw `head_T` from symbols outputting `T` (epoch alphabet), tails from terminals of `T`, Dc below `C_T`; no `gene_root` write.
- `809949ba` typed_vary: mutation/IS/RIS/inversion/crossover per `(gene, domain, type)`; whole-gene moves from the homologous-pair table only (no root carriage); the per-type write-back for cleanse with promotion's enumerated domain; root draw on stream 40 goes.
- `0b321a52` keep; add the wrong-output-in-`head_T` defect to the fixture.
- Paging (`CR_multityped_gep.md` §10c) is unchanged in kind; the page planner takes the rev-4 widths from the fuller bound table.

## For Andrew

Nothing blocks. One default I set: `N`, the program size a typed layout declares, is a layout parameter like `h` is for SR; pick it per kingdom when the layouts are built.
