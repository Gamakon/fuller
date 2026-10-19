# Multi-typed GEP revision 3 — implementation impact

## What changed

The design now has one `head_T` and one `tail_T` per output type, rather than a shared head filtered into type frames. Head lengths are fixed layout parameters; with `H = sum_T h_T`, the tail bound is `min(H*a_T + [T=R], H*(n_max-1)+1)`. This is the chosen design because it preserves Karva locality and homologous crossover within a type domain. With one type it is exactly classical Karva.

Ferreira's names are now explicit: population-wide shared definitions are typed, immutable **ADF genes** with stable ids and strictly decreasing ADF levels; the rewritten remainder/linker is the **homeotic gene**. A typed ADF reference is a genic terminal. A chromosome may instead use a fixed typed linker. Several homeotic genes are several main programs in a multicellular system.

## Built-step changes

### phylu

- `65a1cbc7`: change the layout/type table from one untyped head plus typed tails to typed heads and typed tails; carry every `h_T`, calculate `H` and the revised `t_T`, and page by `(gene, domain, type, offset)` for heads too.
- `f279b306`: make the device walk select `head_T` then `tail_T` directly. Its per-type cursors remain, but frame construction must not filter a shared head. Preserve defect reporting and add the cross-domain-locality parity case.
- `34bae3a2`: generate each `head_T` only from symbols outputting `T`; use the `H` bound for each tail and retain the queue simulation/100k closure test.
- `809949ba`: scope head mutation, IS/RIS, inversion, and crossover to the same output-type head domain. Update random/operator closure and seam-page tests. Add fixed/evolved typed linker genes, typed genic terminals, and multiple main-program support.
- `0b321a52`: retain the function-in-tail defect, but extend the layout checker/fixture to reject a head symbol whose output differs from that head's `T` and to cover the new per-type heads.

### fuller

- `74b03c73`: retain the kingdom contract, but replace its shared-head layout and tail calculation with `h_T`, `H`, and the new bound; validate typed heads.
- `0d13d9c1`: revise host storage and FIFO expression from a filtered common head to actual `head_T` domains; keep per-type Dc cursors and no-repair defects.
- `f4f1dde8`: update closure/bound generators and add tests that a `head_T` edit cannot change another frame, same-type crossover is homologous, and one-type equality remains exact.
- `6c809bd4`: update the DuckDB fixture/query to the `domain_type`-keyed heads and test two genes/two chromosomes as well as the two-type tree.
- `569e27a3`: reverse the terms and model: shared definitions become ADF genes; the rewritten typed linker becomes the homeotic gene. Implement typed ADF terminals, stable `adf_id`, immutable root types, lower-level-only ADF calls, fixed-linker alternative, and multiple homeotic main programs.
- `a1eeaa75`: retain the function-in-tail defect; add the wrong-output-in-`head_T` defect and adjust expected layout diagnostics.

The same terminology change must be made in fuller's `docs/PLAN_population_dag.md`, `src/population_dag.rs`, and `src/homeotic.rs`: names that presently mean a shared definition/row/tail become ADF names; “homeotic gene” is reserved for the evolved linker.
