# Advisor review: Multi-typed GEP revision 3 (a709001b)

**Verdict: sound at the gene, not yet at the chromosome, and the tail bound as written makes WGSL infeasible in width.** I could not build an invalid tree, an exhausted tail, a cycle or a Dc misuse from any single-gene operator in §5: every frame is a prefix stream of one type, `t_T` over-provisions, `q_T` ≤ expressed T leaves ≤ `t_T`. Per-type heads are the right choice over a shared head: a type-T edit lands in the k-th T demand, same-type crossover is homologous, one type is exactly Karva. Three things block building.

## Findings, by severity

**1. The bound uses `H = Σ h_T`, so width scales with the type count squared.** WGSL, 204 types, `h_T = 32`: `H = 6528`, `t_T = 3H+1 = 19,585` per type, about 8 M words (32 MB) per gene, 500× the CR's 16,134. Fix: count demand by producer: `t_T = [T=R] + Σ_U h_U · a_{T,U}`, where `a_{T,U}` is the most T-arguments of any function outputting U (min with `Σ_U h_U (n_max−1)+1`). Same proof, sparse signatures make it small. Walk scratch is `H·n_max+1`, not `h·n_max+1`.

**2. Genic terminals escape their type when a gene's root moves.** §6 names genic terminal `G_i` by gene id and types it by that gene's root; the CR resolution moves `gene_root` with the record in gene transposition. Genes `g0:R, g1:A`, homeotic `head_R = [link:R(R,A), G0, …]`; transpose g0↔g1 and `G0` now expresses an A tree in an R slot. Fix: root is a property of `(layout_id, gene_id)`; whole-gene moves only between equal-root ids; `g_T` then is a layout constant.

**3. An ADF gene cannot hold an arbitrary folded subtree.** Fold writes a shared DAG node into a gene with fixed `h_T`; a subtree with more T functions than `h_T` has no encoding, and the design forbids rejection. ADFs are immutable, so store them as expressed trees (node rows with root type and level), not Karva genes; only the homeotic gene needs a layout.

**4. Underspecified before build:** the homeotic head alphabet when the whole remainder is rewritten into it (all functions, or linkers only); whether normal genes may carry ADF terminals; that a population ADF never references a chromosome-local gene; the relevel/write-back rule into per-type heads for fold, cleanse and snap (the CR's `j_T` rule, absent here); the typed `?_T` for ADF-bearing types.

**5. WGSL will be valid, not yet good.** 129 of 204 duals have no terminal (7aa21d88), so the kingdom check fails today. Equal `h_T` wastes most of the budget: with statement nesting in types, `h_stmt` is the program-length cap. Allocate `h_T` from the kingdom's demand matrix (expected T demands from the root at the target program size), then fix it.

Minor: "an edit in `head_T` affects only T's frame" is true of bytes only; it still changes which rows other frames express, as in Karva. The SQL needs `symbol` to be an id, not a spelling, for polymorphic rows.
