# Multi-typed GEP: a head and tail for every output type

## Context

**Objective.** Multi-typed gene expression programming in which every expressed node has the type required by its parent's argument slot, for every gene and every permitted variation. Validity is a property of the layout: there is no repair, rejection, substitution, or fallback symbol anywhere.

**Approach.** A gene has a separate `head_T` and `tail_T` for every output type `T`. A slot requiring `T` reads the next unused member of `T`'s frame: `head_T`, then `tail_T`. `head_T` contains only symbols whose output is `T`; `tail_T` contains only terminals whose output is `T`. This preserves type correctness *and* makes mutation and crossover local to the output-type domain. Chromosomes are multigenic: typed normal genes under a typed linker that is Ferreira's homeotic gene; population-wide shared subgraphs are typed ADFs.

**Unique feature.** Every bound is derived from the kingdom's ordered signatures and proved, never tuned to a buffer: the geneframe is built on the GPU and paged across buffers (CR_multityped_gep §10c) precisely to hold whatever the bounds say. Width is a consequence of the design, never an input to it.

**Status.** Design, revision 4. Revision 3 was built host-side in fuller (`74b03c73`..`a1eeaa75`) and device-side in phylu (`65a1cbc7`..`0b321a52`) with a shared head; the advisor review (`docs/reviews/2026-10-20_multityped_gep_fable.md`) found five defects, fixed here. The note `docs/reviews/2026-10-20_multityped_gep_fable_rev4.md` lists what each built commit must change.

**Why.** Fallback leaves hide bugs and flatten the search; a shared head loses Karva locality between types; a bound on the type count squared and a head budget guessed per type cap the programs a kingdom can express. The per-producer bound and the program-size head allocation below are exact, so nothing is capped that the design did not intend.

## 1. Kingdom contract

`symbol_outputs(kingdom_id, symbol_id, spelling, output_type)` is a unified relation over functions, terminals and reference terminals (§6). Every `symbol_id` has exactly one output type; a polymorphic spelling has one `symbol_id` per output type. Every gene row stores a `symbol_id`, never a spelling.

```
argument_signature(kingdom_id, symbol_id, arg_index, required_type)
PRIMARY KEY (kingdom_id, symbol_id, arg_index)
```

`arg_index` is contiguous from zero for each function. A terminal has no signature rows. Thus `left:S(S,I)` has `(left,0,S)` and `(left,1,I)`: argument order is data, not an implication of type counts.

The kingdom check establishes exactly one output row per symbol; ordered, contiguous signatures on functions only; and at least one terminal for every root or demanded type. It derives, from the same rows:

- `a_{T,U}`: the most arguments of type `T` any one function outputting `U` takes (zero if no function outputting `U` asks for `T`);
- `a_T = max_U a_{T,U}`, and `n_U`: the largest arity among functions outputting `U` (one if there are none);
- `reachable(R)`: the types reachable from a root `R` through signatures.

## 2. Gene layout, heads, and bounds

Each row has the domain-local key:

```
(chromosome_id, gene_id, domain, domain_type, domain_seq_id)
```

`domain` is `head`, `tail`, or `dc`; `domain_type` is the output type for every head, tail, and Dc domain; and `domain_seq_id` starts at zero within that domain. A head row stores a symbol outputting `domain_type`; a tail row stores a terminal outputting it; a Dc row stores `dc_index`, never a symbol. `gene_id` is a fixed gene boundary, not a breadth-first position.

### 2.1 Head lengths from the program size

A layout declares, for every gene id, its root type `R` and either a **program size** `N` (the number of expressed functions the layout must hold) or explicit head lengths. From `N`:

```
h_T = [T = R] + (N - 1) * a_T        over T in reachable(R)
```

This is exact. In level order the last `T` function has before it at most the root and the `T` slots opened by the other `N - 1` functions, each at most `a_{T,U} <= a_T` of them, so every program of at most `N` functions fits every `head_T` whatever its type mix, and the chain of `N` functions each giving its `T` child last attains the bound. It is the evolvability criterion: a correct program of size `N` is reachable by the write-back of §5 in every kingdom, and no type is capped by a guessed budget (in WGSL the statement types get `h >= N`, so statement nesting is never the program-length cap). With explicit `h_T` the check reports the largest `N` the heads guarantee. Fixed domain lengths make homologous crossover meaningful; `h_T` is a layout parameter, never evolved.

### 2.2 The per-producer tail bound

At most `h_U` functions are expressed from `head_U` (each row is read at most once; tail rows and reference terminals open no slots). The demand for `T` is therefore bounded by its producers and by the leaf count:

```
P_T = [T = R] + sum_U h_U * a_{T,U}
L   = 1 + sum_U h_U * (n_U - 1)
t_T = max(0, min(P_T, L + h_T) - h_T)     tail_T length
d_T = min(P_T, L)                        Dc_T length (the most expressed T leaves)
```

`P_T` counts every `T` slot any expressed function can open; `L` counts every leaf any expressed program can have, and the `T` demands are the `T` leaves plus at most `h_T` expressed `T` functions, so `D_T <= min(P_T, L + h_T)`. The first `h_T` of those demands read `head_T`, hence `t_T`. The expressed `?_T` are `T` leaves, hence `d_T`. Both are exact: each term is attained by a program the layout can hold. Do not create a non-root `tail_T` when `P_T = 0`; a root tail exists even when its head is empty.

With one type `R`, `a_{R,R} = n_max`, `P = 1 + h * n_max`, `L = 1 + h * (n_max - 1)`, so `t = d = h * (n_max - 1) + 1`: the layout is the physical Karva head and tail, and the decoder below is exactly classical Karva. With several types the bound is a sum over producers, so sparse signatures make it small, and it is never traded against width: the layout is allocated in full and paged.

Compared with revision 3 (`min(H * a_T + [T = R], H * (n_max - 1) + 1)` with `H = sum_T h_T`), this bound is not larger for any kingdom and is exact for every one, including `t_T <= t_T^{rev3} - h_T`.

## 3. Deterministic expression

Start FIFO `pending` with `(R, depth 0)`. Keep `k_T = 0` for every type. On dequeue of a `T` slot, select row `T[k_T]` from `head_T` followed by `tail_T`, increment `k_T`, and append the selected symbol's ordered-signature slots. Stop at an empty queue. The bound prevents tail exhaustion and frame membership proves the selected symbol has type `T`. Reference terminals (§6) have no signature rows and so are leaves; their meaning (a gene or an ADF) is resolved after the walk, by id.

### Executable DuckDB bound and expression queries

Relations: `functions`, `terminals`, `reference_terminals` (each `kingdom_id, symbol_id, spelling, output_type`; references add `kind, target_id`), `argument_signature` as in §1, `layouts(layout_id, kingdom_id, program_size)`, `layout_genes(layout_id, gene_id, root_type)`, `layout_heads(layout_id, gene_id, type, h)`, `genes(chromosome_id, gene_id, kingdom_id, layout_id)`, and `geneframe` keyed as in §2 with `symbol_id` and `dc_index`.

The bound query derives `P_T`, `L`, `t_T` and `d_T` per gene of a layout from the same rows the decoder reads:

```sql
CREATE OR REPLACE VIEW symbol_outputs AS
SELECT kingdom_id, symbol_id, spelling, output_type FROM functions
UNION ALL
SELECT kingdom_id, symbol_id, spelling, output_type FROM terminals
UNION ALL
SELECT kingdom_id, symbol_id, spelling, output_type FROM reference_terminals;

-- a_{T,U}: the most arguments of type T any one function outputting U takes.
CREATE OR REPLACE VIEW producer_demand AS
SELECT f.kingdom_id, a.required_type AS t, f.output_type AS u, max(c.n) AS a_tu
FROM functions AS f
JOIN (SELECT kingdom_id, symbol_id, required_type, count(*) AS n
      FROM argument_signature GROUP BY kingdom_id, symbol_id, required_type) AS c
  ON c.kingdom_id = f.kingdom_id AND c.symbol_id = f.symbol_id
JOIN argument_signature AS a
  ON a.kingdom_id = c.kingdom_id AND a.symbol_id = c.symbol_id AND a.required_type = c.required_type
GROUP BY f.kingdom_id, a.required_type, f.output_type;

-- n_U: the largest arity of a function outputting U.
CREATE OR REPLACE VIEW producer_arity AS
SELECT f.kingdom_id, f.output_type AS u, max(coalesce(s.n, 0)) AS n_u
FROM functions AS f
LEFT JOIN (SELECT kingdom_id, symbol_id, count(*) AS n FROM argument_signature GROUP BY kingdom_id, symbol_id) AS s
  ON s.kingdom_id = f.kingdom_id AND s.symbol_id = f.symbol_id
GROUP BY f.kingdom_id, f.output_type;

-- Per gene of a layout: P_T, L, t_T and d_T from its heads h_T.
CREATE OR REPLACE VIEW layout_bounds AS
WITH heads AS (
  SELECT lh.layout_id, lh.gene_id, l.kingdom_id, lg.root_type, lh.type, lh.h
  FROM layout_heads AS lh
  JOIN layout_genes AS lg ON lg.layout_id = lh.layout_id AND lg.gene_id = lh.gene_id
  JOIN layouts AS l ON l.layout_id = lh.layout_id
),
leaves AS (
  SELECT hd.layout_id, hd.gene_id, 1 + sum(hd.h * greatest(coalesce(pa.n_u, 1) - 1, 0)) AS l
  FROM heads AS hd
  LEFT JOIN producer_arity AS pa ON pa.kingdom_id = hd.kingdom_id AND pa.u = hd.type
  GROUP BY hd.layout_id, hd.gene_id
),
demand AS (
  SELECT t.layout_id, t.gene_id, t.type,
         (CASE WHEN t.type = t.root_type THEN 1 ELSE 0 END)
         + coalesce(sum(u.h * pd.a_tu), 0) AS p
  FROM heads AS t
  LEFT JOIN heads AS u ON u.layout_id = t.layout_id AND u.gene_id = t.gene_id
  LEFT JOIN producer_demand AS pd ON pd.kingdom_id = t.kingdom_id AND pd.t = t.type AND pd.u = u.type
  GROUP BY t.layout_id, t.gene_id, t.type, t.root_type
)
SELECT hd.layout_id, hd.gene_id, hd.type, hd.root_type, hd.h, d.p, lv.l,
       greatest(0, least(d.p, lv.l + hd.h) - hd.h) AS t,
       least(d.p, lv.l) AS d
FROM heads AS hd
JOIN demand AS d ON d.layout_id = hd.layout_id AND d.gene_id = hd.gene_id AND d.type = hd.type
JOIN leaves AS lv ON lv.layout_id = hd.layout_id AND lv.gene_id = hd.gene_id;
```

The expression query implements the complete FIFO walk. `used` supplies the per-type rank before the current slot is added; `parent_step` and `arg_index` retain ordered edges.

```sql
WITH RECURSIVE
frame_source AS (
  SELECT g.chromosome_id, g.gene_id, g.kingdom_id, g.domain_type AS output_type,
         CASE g.domain WHEN 'head' THEN 0 ELSE 1 END AS tail_part, g.domain_seq_id, g.symbol_id
  FROM geneframe AS g JOIN symbol_outputs AS o
    ON o.kingdom_id = g.kingdom_id AND o.symbol_id = g.symbol_id
   AND o.output_type = g.domain_type
  WHERE g.domain IN ('head', 'tail')
),
frames AS (
  SELECT *, row_number() OVER (
    PARTITION BY chromosome_id, gene_id, kingdom_id, output_type
    ORDER BY tail_part, domain_seq_id
  ) - 1 AS frame_k FROM frame_source
),
walk(chromosome_id, gene_id, kingdom_id, step, pending, used,
     node_type, frame_k, symbol_id, depth, parent_step, arg_index) AS (
  SELECT ge.chromosome_id, ge.gene_id, ge.kingdom_id, 0,
         list_value(struct_pack(required_type := lg.root_type, depth := 0::BIGINT,
                                parent_step := NULL::BIGINT, arg_index := NULL::BIGINT)),
         []::VARCHAR[], NULL::VARCHAR, NULL::BIGINT, NULL::BIGINT,
         NULL::BIGINT, NULL::BIGINT, NULL::BIGINT
  FROM genes AS ge JOIN layout_genes AS lg ON lg.layout_id = ge.layout_id AND lg.gene_id = ge.gene_id
  WHERE ge.chromosome_id = ?
  UNION ALL
  SELECT w.chromosome_id, w.gene_id, w.kingdom_id, w.step + 1,
         list_concat(list_slice(w.pending, 2, len(w.pending)), coalesce(kids.slots, list_value())),
         list_append(w.used, q.slot.required_type), q.slot.required_type,
         len(list_filter(w.used, x -> x = q.slot.required_type)), f.symbol_id,
         q.slot.depth, q.slot.parent_step, q.slot.arg_index
  FROM walk AS w CROSS JOIN LATERAL (SELECT list_extract(w.pending, 1) AS slot) AS q
  JOIN frames AS f ON f.chromosome_id = w.chromosome_id AND f.gene_id = w.gene_id
    AND f.kingdom_id = w.kingdom_id AND f.output_type = q.slot.required_type
    AND f.frame_k = len(list_filter(w.used, x -> x = q.slot.required_type))
  LEFT JOIN LATERAL (
    SELECT list(struct_pack(required_type := a.required_type, depth := q.slot.depth + 1,
                            parent_step := w.step + 1, arg_index := a.arg_index) ORDER BY a.arg_index) AS slots
    FROM argument_signature AS a WHERE a.kingdom_id = w.kingdom_id AND a.symbol_id = f.symbol_id
  ) AS kids ON TRUE WHERE len(w.pending) > 0
)
SELECT w.gene_id, w.step, w.node_type, w.frame_k, o.spelling, w.depth, w.parent_step, w.arg_index
FROM walk AS w JOIN symbol_outputs AS o ON o.kingdom_id = w.kingdom_id AND o.symbol_id = w.symbol_id
WHERE w.step > 0 ORDER BY w.gene_id, w.step;
```

The decoder reports a kingdom/layout defect when a frame row is missing; it never repairs. The partition deliberately includes chromosome, gene, and type.

Run on 2026-10-20 in DuckDB 1.5.6 on the two-type fixture `left:S(S,I)`, terminals `s:S`, `i:I`, and the polymorphic spelling `z` as `z:S` and `z:I`, with `h_S = 2`, `h_I = 1`, root `S`: the bound query gives `P_S = 3, P_I = 2, L = 3, t_S = t_I = 1, d_S = 3, d_I = 2`; the one-type SR layout `h = 7`, `n_max = 2` gives `t = d = 8`. The walk gives, for `head_S = (left, s)`, `head_I = (i)`: `left(s, i)` as steps `(S,0,left)`, `(S,1,s)`, `(I,0,i)`; for a second gene `head_S = (left, left)`, `tail_S = (s)`, `tail_I = (z)`: `left(left(s, z), i)`; for a second chromosome with the ADF reference `A7:I` in `head_I`: `left(s, A7)`. The physical rows read are in their own domains, not an inter-type shared stream.

## 4. Constants (GEP-RNC, per type)

A constant-capable type `T` (a kingdom property: it has a literal domain) has placeholder terminal `?_T` and `Dc_T` of length `d_T`. `?_T` may sit in `head_T` or `tail_T`; only an **expressed** placeholder consumes the next `Dc_T` entry and advances `q_T`, and `q_T <= d_T` by §2.2. `C_T` is append-only for a population epoch; existing indices are immutable. Dc mutation, recombination, inversion, and transposition stay in their matching `(gene_id, T, dc)` domain and select valid `C_T` indices. Whether `T` has ADFs (§6) is independent of whether it has `?_T`: both are terminals of `T` and coexist in its alphabet.

## 5. Variation: domains are invariant

Every operator draws from a domain enumerated from the layout before the draw, so every child is valid by construction. An operator whose domain is empty for a layout is absent from that layout's schedule, the way collapse is absent for a type without `?_T`; nothing is drawn and then replaced.

Mutation selects any symbol of the epoch alphabet (§6) outputting `T` in `head_T`, a terminal outputting `T` in `tail_T`, or a valid index in `Dc_T`. Inversion and tail/Dc transposition stay within one exact `(gene, domain, type)` domain. IS and RIS may copy a head segment into the *same* `head_T` only (RIS starts at a function); no tail or Dc material enters a head. One-/two-point and gene recombination exchange homologous coordinates only: same kingdom, layout id, gene id, domain, output type, and local positions. All operations preserve domain lengths and alphabets. An edit in `head_T` changes only `T`'s bytes; as in Karva it may change which rows other frames express, never their validity.

**Whole-gene moves.** The root type and the domain lengths of a gene are properties of `(layout_id, gene_id)`, never of the record: no `gene_root` word travels with a gene. Two gene ids are homologous when the layout gives them the same root and the same `h_T`. Gene recombination exchanges the same gene id between parents of one layout (always homologous); gene transposition and the whole-gene parts of one-/two-point recombination move a record only between homologous gene ids, the ordered pairs of which the layout enumerates. A genic terminal (§6) therefore always names a gene of its own type.

**Write-back (fold, cleanse, snap).** A rewrite of an expressed tree is encoded by the walk of §3 in reverse: walk the new tree in level order, number the `T` nodes `0, 1, 2, ...`, and let `j_T` be one more than the rank among `T` nodes of the last `T` function (zero with none). `head_T` holds the first `j_T` `T` nodes, `tail_T` the rest in order; `Dc_T` is rebuilt in the order the expressed `?_T` appear; rows the walk does not read keep their previous contents (Karva's non-coding region, deterministic). The rewrite is encodable exactly when `j_T <= h_T` for every `T`. Removal-only rewrites (fold replaces a subtree by a reference, collapse by `?_T`, snap by a constant) delete nodes without changing any remaining depth, so no rank grows and they always fit. Promotion lifts a subtree one level and can grow a rank: with one type, `P(Q(Z(a,b,c)), W(F(u,v)))` has `j = 5`, and promoting `Q` to `Z` gives `P(Z(a,b,c), W(F(u,v)))` with `j = 7`. Promotion's domain on a gene is therefore the set of same-type children whose promoted tree fits every `h_T`, enumerated before the draw (the CR's rule, now per type rather than `sum_T j_T <= vhead`).

## 6. Typed linkers, ADFs, and multicellular programs

**Epoch alphabet.** The symbols a gene may hold are the kingdom's symbols plus two kinds of typed **reference terminal**, each a row of `symbol_outputs` with no signature:

- a **genic terminal** `G_g` for every normal gene id `g` of the layout, outputting `root(layout_id, g)`; `g_T` is the number of them of type `T`, a layout constant;
- an **ADF reference** `A_d` for every ADF `d` of the population epoch, outputting `d`'s declared root type.

Genic terminals are valid only in a homeotic gene; ADF references are valid in any `head_T` or `tail_T` of their type, normal or homeotic, so a reference propagates by ordinary mutation and fold can rewrite a shared subtree wherever it occurs. The ADF set is append-only within an epoch, ids are never reused, and an ADF with zero referrers is retired only at an epoch boundary, so every reference in a population names an ADF that exists.

**The homeotic gene.** A chromosome's normal genes are combined by a typed linker, which is Ferreira's evolved homeotic gene: a typed gene with the `head_T`/`tail_T` layout of §2 whose declared root is the chromosome phenotype type and whose alphabet is the epoch alphabet including the chromosome's genic terminals (any symbol of `T` in `head_T`, any terminal of `T` in `tail_T`). Its head alphabet is the whole function set, not linking functions only, because fold rewrites the whole remainder of a program into it and the remainder holds ordinary functions and terminals; a linking function is simply a kingdom function whose signature consumes gene root types. Expression invokes each selected genic terminal's gene. A **fixed linker** is a homeotic gene the layout marks immutable (`head_R = (link)`, tails the genic terminals in gene order): every operator's domain on it is empty. Several homeotic genes mean several main programs (a multicellular system): each has a stable main-program id, declared phenotype type, and linked normal-gene set; evaluate each independently, retaining their ordered result tuple. Normal genes never hold genic terminals, so gene references cannot cycle.

**ADFs.** Population-wide definitions found by share-equivalence are Ferreira's **ADFs**. An ADF is immutable, so it has no genotype to vary: it is stored as its expressed tree, `adf(epoch, adf_id, root_type, level)` with node rows `adf_node(adf_id, node_seq, symbol_id, parent_seq, arg_index, dc_index)` in level order, a `?_T` node carrying its `C_T` index. This is the population DAG's node storage (`src/population_dag.rs`), so a folded subtree of any size has an encoding; a Karva gene with fixed `h_T` would not hold a subtree with more `T` functions than `h_T`, and the design forbids rejection. An ADF body holds kingdom symbols and ADF references of strictly lower level only: never a genic terminal (a genic terminal means a different tree in every chromosome; an ADF is one tree for the population). Fold turns each selected shared DAG node into an ADF with a stable `adf_id`, root type and level, and rewrites every chromosome holding that subtree by the write-back of §5 (a removal-only rewrite, so it always fits), replacing the subtree's root with `A_d` in whichever gene holds it; a subtree that reaches a normal gene through `G_g` is replaced at its root in the homeotic gene. Unfold inlines `A_d` by its tree. Stable ids, root types, immutability, and strict level decrease give persistent references and acyclicity. (Were ADFs ever varied, the equally valid encoding is a Karva gene under its own layout id with `h_T` the subtree's `T` function count; it is not built.)

Required code/document rename (not performed by this document-only change): in `docs/PLAN_population_dag.md`, replace "homeotic chromosome/row/tail/definition" when it denotes a population-wide shared definition with "ADF gene/row/set/definition"; retain "homeotic gene" only for the rewritten chromosome linker. In `src/population_dag.rs`, rename `is_homeotic`, `homeotic`, `homeotic_new`, related fields/comments/tests, and memo-selection terminology to `is_adf`, `adfs`, `adf_new`, and ADF. In `src/homeotic.rs`, rename the share-finder fold product, `FoldedChromosome` head/tail documentation, href vocabulary, and tests from "homeotic tail/slot" to typed ADFs/`adf_id`; reserve `homeotic` for the new linker implementation. The equivalent phylu `GENEREF`, export, checkpoint, arena, and level-schedule names must make the same distinction.

## 7. Implementation and tests

1. Kingdom/layout checks: ordered signatures, terminal coverage, `a_{T,U}`, `n_U`, `h_T` from `N` (or `N` from explicit `h_T`), `P_T`, `L`, `t_T`, `d_T`, per-domain alphabets, homologous gene-id pairs, `g_T`. The check prints the bound table per kingdom (SR, regex, safewgsl) with `N` and the gene width; the WGSL figures are measured here, not estimated.
2. Bound exactness: for every kingdom and `h_T`, construct the program attaining `P_T`, the one attaining `L + h_T`, and the chain attaining `h_T` from `N`; assert the walk reads the last frame row and no further. 100,000 random genes per kingdom: record maximum `k_T` and `q_T`, below `h_T + t_T` and `d_T`; include expressed-only, interleaved placeholders. One type: `t = d = h * (n_max - 1) + 1` and `h = 1 + (N - 1) * n_max`.
3. Host and GPU closure: 100,000 random genes and children per operator for SR, regex, and WGSL; every node matches its slot type and neither decoder substitutes; an operator with an empty domain is absent, never a no-op draw.
4. SQL: run §3's two queries in `duckdb :memory:` on the two-type fixture with the polymorphic spelling, two genes, two chromosomes and an ADF reference, asserting the bound rows and the stated trees; the host decoder equals the query row for row.
5. Locality/Karva: edits in `head_T` do not alter another frame's bytes; same-type crossover is homologous; one-type genes decode byte-for-byte as classical Karva.
6. Gene moves: gene transposition and whole-gene recombination between non-homologous ids are not in the enumerated domain; after any sequence of moves every genic terminal expresses a tree of its own type; a layout with one gene per root has no transposition pairs and the schedule says so.
7. Write-back: removal-only rewrites never grow any `j_T`; the promotion example above is excluded from the domain for `h < 7` and included for `h >= 7`; fold → write-back → unfold is value-preserving on 100,000 random chromosomes.
8. ADFs: a folded subtree larger than every `h_T` is stored and referenced; an ADF body holding a genic terminal, a reference to a non-lower level, or a reference of the wrong type is a defect; `A_d` in a normal gene and in a homeotic gene both express; a retired ADF has zero referrers.
9. Linkers: fixed and evolved homeotic genes, genic-terminal type/count checks, multiple homeotic main programs, the full function set in the homeotic head.

## 8. Build order

1. Kingdom contract: `symbol_id`, `a_{T,U}`, `n_U`, reachability, the bound table printed per kingdom; the safewgsl terminal rows of `docs/PLAN_wgsl_terminals.md` so the check passes on the live kingdom.
2. Layouts: per-type heads from `N` or explicit `h_T`, roots and lengths by `(layout_id, gene_id)`, homologous pairs; host decoder; the DuckDB fixture with both queries.
3. Host operators, including whole-gene moves by homologous id and the per-type write-back; the locality, closure, bound-exactness, and one-type Karva tests.
4. GPU per-type heads and cursors, paged layouts, parity; remove all fallback bindings and every `gene_root` word.
5. Reference terminals, the homeotic gene over the epoch alphabet, fixed linkers, ADFs as trees in the population DAG with stable ids/levels, fold/unfold, and the ADF rename across fuller and phylu.
6. Move SR and regex, then WGSL; re-encode saved genomes and run round trips.

## Revisions (2026-10-20)

- Replaced the shared head with a typed head and tail for every output type; made it the chosen design because it restores per-domain locality and homologous crossover.
- Derived the multi-head tail bound using `H = sum_T h_T`, specified fixed head-length selection, and retained exact one-type Karva reduction.
- Adopted Ferreira terminology: shared population definitions are ADF normal genes; the rewritten linker remainder is the homeotic gene.
- Specified typed genic terminals, evolved/fixed linkers, multicellular multiple-homeotic-program systems, and the share-equivalence-to-ADF/homeotic transformation.

## Revisions (2026-10-20, advisor)

- Finding 1 (bound on the type count squared): the tail bound is per producer, `P_T = [T = R] + sum_U h_U * a_{T,U}`, with the leaf bound `L`; `t_T = max(0, min(P_T, L + h_T) - h_T)` and `Dc_T` length `d_T = min(P_T, L)`. Both exact, both attained, one type is Karva unchanged. The bound SQL is executable and was run.
- Finding 2 (genic terminals escape their type): root and domain lengths belong to `(layout_id, gene_id)`; whole-gene moves only between homologous gene ids, enumerated by the layout; `gene_root` words removed; `g_T` is a layout constant.
- Finding 3 (an ADF cannot hold an arbitrary subtree): ADFs are immutable expressed trees in the population DAG's node storage, never Karva genes; only genes that vary have a layout.
- Finding 4 (underspecified): the homeotic head holds the whole function set plus reference terminals; ADF references are valid in normal genes; an ADF never references a genic terminal; the write-back rule is per type (`j_T <= h_T`), removal-only rewrites always fit and promotion's domain is enumerated; `?_T` is independent of ADFs.
- Finding 5 (valid, not good): `h_T = [T = R] + (N - 1) * a_T` from the layout's program size `N`, exact and attained, so every program of `N` functions fits whatever its type mix; explicit `h_T` remains for the classical SR layout. Width is allocated in full and paged, never an input.
- `symbol_id` replaces spellings in every gene row and in the SQL; the kingdom check prints the WGSL bound table rather than this document estimating it.

## Implementation notes

Host reference, fuller `ca816f1e`..`ff7fccc8` and the defect tests that follow. Where the design left a choice, this is what was built and why.

1. **`a_T` for head allocation is taken over the producers reachable from the root.** `h_T = [T = R] + (N - 1) * a_T` uses `a_T = max_U a_{T,U}` over `U` in `reachable(R)`, not over every `U` in the kingdom. A producer no slot can reach is never expressed, so counting it would only widen the head (a 4-argument function outputting an unreachable `B` would otherwise cost `3 * (N - 1)` head rows of `I` in every `S` gene). The bound stays exact; `KingdomContract::guaranteed_program_size` is the inverse and is asserted for every `N` in the tests. `P_T` and `L` already sum over the heads actually present, so they were exact.
2. **`d_T` is `min(P_T, L)` for every type; a `Dc_T` exists only for a constant-capable type.** The bound query and `KingdomContract::bounds` agree row for row (the DuckDB test compares 25 rows). `GeneBounds::width` counts `d_T` only where the type has `?_T`.
3. **Physical order of a gene** is every head, then every tail, then every Dc, each in frame order (the root's type first, then ascending type code). A one-type gene is therefore exactly Karva's head, tail and Dc, which the tests compare with an independent classical decoder (`h = 1 + (N - 1) * n_max`, `t = d = h * (n_max - 1) + 1`, width `h + 2t`).
4. **Write-back places the k-th `T` node of the level order at row `k` of `T`'s frame** (`head_T` for `k < h_T`, then `tail_T`), not "the first `j_T` nodes in the head and the rest in the tail". The two agree when the tree has no more than `j_T` nodes of `T`; when it has more, the nodes between `j_T` and `h_T` are terminals and must sit in `head_T` rows `j_T .. h_T - 1`, because the decoder reads `head_T` to its end before `tail_T`. A tree is encodable exactly when `j_T <= h_T` for every `T` and its nodes and placeholders fit the tails and Dcs; otherwise the reported defect names the type and the missing rows.
5. **Validation is of the whole gene, read or not.** A head cell whose output is not its domain's type (`ForeignSymbol`), a tail cell that is a function (`FunctionInTail`), a symbol outside the kingdom and a Dc word at or past `C_T` are defects wherever they sit, because the domain alphabets define a valid gene; the planted-defect tests put one in 100,000 genes per kingdom.
6. **Reference terminals are a re-checked contract.** `Epoch::new` builds the epoch alphabet as the kingdom's rows plus `G_g` for every normal gene id (typed by the layout's root) and `A_d` for every declared ADF, and runs the kingdom check again, so a reference is an ordinary terminal with no signature and every bound is the kingdom's. Genic terminals are valid only in a homeotic gene that links their gene; an unlinked genic terminal is a defect even in a cell the walk never reads.
7. **A fixed linker's unread rows are canonical**: `head_R = (link)`, the genic terminals of the linked genes in gene order in `tail_R` / `tail_T`, every other tail row the lowest terminal of its type, every Dc word 0. Any other word is `FixedLinkerAltered`.
8. **Fold is by structural equality of expressed trees** (symbols and `C_T` indices), top-down, replacing each match by `A_d` through the write-back; a removal-only rewrite always fits, and a write-back that fails would be reported, not skipped. Unfold at the phenotype level is `Phenotype::inlined`; unfolding back into a gene is encodable exactly when `typed_gep::fits` says so.
9. **Not done in this series:** the §6 rename of "homeotic chromosome/row/tail" to "ADF" in `docs/PLAN_population_dag.md`, `src/population_dag.rs` and `src/homeotic.rs`, and the `docs/PLAN_wgsl_terminals.md` rows (owned by the WGSL terminals work). The population DAG still stores shared definitions under the old name.

Kingdom check figures (`cargo test --release --features wgsl --lib the_kingdom_reports_print -- --nocapture`, program size `N`, heads from `h_T = [T = R] + (N - 1) * a_T`), measured on 2026-10-20:

| kingdom | N | roots | types reachable | L | gene width (words) | widest type (h, P, t, d) |
|---|---|---|---|---|---|---|
| SR | 8 | F | 1 | 16 | 47 | F: 15, 31, 16, 16 |
| SR | 32 | F | 1 | 64 | 191 | F: 63, 127, 64, 64 |
| regex | 8 | Pattern | 4 | 30 | 177 | Char: 14, 43, 29, 30 |
| regex | 32 | Pattern | 4 | 126 | 753 | Char: 62, 187, 125, 126 |
| safewgsl | 8 | `Wgsl(107)` / `Wgsl(186)` | 173 / 174 | 3958 / 4036 | 58,090 / 58,211 | `Wgsl(33)`: 21, 1379, 1358, 1379 |
| safewgsl | 32 | `Wgsl(107)` / `Wgsl(186)` | 173 / 174 | 17,518 / 17,860 | 257,242 / 257,771 | `Wgsl(33)`: 93, 6107, 6014, 6107 |

The safewgsl width is a consequence of `N` and of 202 types with up to 4-argument functions, not a tuned buffer: it is what the paged geneframe (CR_multityped_gep §10c) has to hold. The older `kingdoms/wgsl` kingdom reads 15,038 words at `N = 8` and 66,590 at `N = 32`.

## Implementation notes (2026-10-20, phylu)

The GPU side of the design, in phylu `src/evolve` (commits `3fd78355` to `1257b834`). Where the design left a choice, this is what was built and why.

### The head's function share and the root cell

A head cell is a function or a terminal of its type. The design did not say how often a function; two settings of the kingdom now do, read by the initialiser and by head point mutation on host and device (`typed_init.rs`, `typed_init.wgsl`, the mutation in `typed_vary.*`), and held in the type directory as one threshold word a type.

1. **`FunctionShare`.** `proportional` is Ferreira's practice: a function is drawn in proportion to the number of functions against terminals that can stand in the cell's type. `half` is a fair coin per cell, which is what the typed initialiser drew until 2026-10-20 and what the untyped engine drew before it. `Fixed(p)` draws a function with probability `p`. The default is `proportional` (`EVOLVE_FUNCTION_SHARE`). `half` is bit-identical to the old draw, so the pinned SR golden keeps its line.
2. **`root_function`.** When on, the root cell of the root type's head (the first symbol of the expression) is drawn from the functions of its type wherever the type has any, in the initialiser and in point mutation. Off is classical GEP. Inversion, transposition and recombination can still place a terminal there. Default off (`EVOLVE_ROOT_FUNCTION=1`).

Expressed size of a randomly drawn gene, host initialiser and the typed walk (`examples/init_size_study.rs`; 2,000 genes, safewgsl 300; SR at head 34, regex and safewgsl at program size 28 and 8):

| kingdom | draw | one-node trees | mean nodes | median | max (bound) |
|---|---|---|---|---|---|
| SR | fair coin | 51.9 % | 2.56 | 1 | 33 (69) |
| SR | coin + root function | 0.0 % | 4.20 | 3 | 33 |
| SR | proportional | 18.1 % | 13.49 | 6 | 48 |
| SR | proportional + root function | 0.0 % | 16.43 | 9 | 49 |
| regex | fair coin | 51.9 % | 2.57 | 1 | 21 (300) |
| regex | coin + root function | 0.0 % | 4.35 | 3 | 29 |
| regex | proportional | 31.4 % | 4.10 | 2 | 40 |
| regex | proportional + root function | 0.0 % | 5.58 | 4 | 44 |
| safewgsl | fair coin | 51.7 % | 9.29 | 1 | 116 (6,087) |
| safewgsl | coin + root function | 0.0 % | 18.95 | 13 | 121 |
| safewgsl | proportional | 51.7 % | 523.22 | 1 | 1,717 |
| safewgsl | proportional + root function | 0.0 % | 1,089.44 | 1,150 | 1,717 |

The classical head draw of a one-type kingdom is the `proportional` row. The fair coin makes about half of all genes a single bare terminal, because the root cell is a terminal half the time. Safewgsl's root type has about as many terminals as functions, so even the proportional draw leaves half its genes a single terminal and makes the rest large; the root function removes the trivial half, and with a proportional share it fills most of the head (median 1,150 nodes), so a fair coin with the root function (mean 19) is the gentler start there. Which setting searches best is not settled by these numbers; the SR 93-law run uses the fair coin on both arms.

### The kernel text does not depend on the layout

A layout of a million words crashed the shader compiler of Metal (`XPC_ERROR_CONNECTION_INTERRUPTED` in the walk kernel). The cause was a shader text that grew with the layout: capacity constants prepended to the text sized private arrays (nine per walk, a dozen per decode, five in the cleanse): private memory a thread proportional to the layout's node bound. The rule since 2026-10-20:

3. **The shipped text is the text.** The walk, init, vary and decode kernels are compiled from the shipped file with nothing prepended. No constant or array in them is derived from a layout.
4. **Scratch is a buffer.** Each kernel keeps what used to be a private array in a storage buffer, a slice of `scratch_stride` words for each invocation of the dispatch (the walk uses its own output record and a small cursor buffer). The capacities (the node bound, the longest segment, the type count) are fields of the uniform. A dispatch covers as many rows as fit 256 MB of scratch, whole rows at a time, so a layout with large per-row scratch is processed in chunks and a small one in one dispatch, as before.
5. **Two contracts are held on the host, with no adapter** (`kernel_contracts.rs`, from naga's reflection of the shipped text): no kernel declares more than 29 bound buffers (Metal's limit for a stage; the decoder had 30, and `op_order` and `op_stack` became one interleaved binding), and every uniform the host fills has the size of the struct the kernel reads (`VP` 22 words, `Island` 22, `InitParams` 12, `DecodeParams` 8, `WalkParams` 8; a mismatch failed every test that opened the pipeline). Each kernel also has a test that its text has no layout constant, is the shipped text, and validates in naga, built against two layouts more than twenty times apart. The regex kernels follow the same rule (mtgep-regex).

The cost is that scratch lives in device memory rather than private memory; the decode of the one-type kingdoms may be slower for it, and the timings of the first run on the new kernels will show how much.

### Checkpoints round-trip every float exactly

A resumed fit must equal an uninterrupted one. Three things were lost on the way through the checkpoint: the device re-froze the HFF column maxima from the resumed population instead of generation 0's; the cohort labels were saved from a stale host copy; and the per-row scores were not saved at all, so the next generation's elites lost theirs (format 3 carries them). Last, `serde_json`'s default parser reads an `f64` back one unit in the last place off, which fed selection at generation 13; the `float_roundtrip` feature makes the read exact for every float field (a test reads awkward `f64` values back bit for bit).

### The untyped engine is gone

SR, regex and the WGSL kingdom run on the typed core alone; nothing is substituted anywhere in phylu. Removed (CR_multityped_gep section 11): `EvolveDevice` and its kernels (`evolve.wgsl`, `decode.wgsl` with its fallback bindings and projection block, `vary.wgsl`); the host copies of the operators (`mutate`, `crossover`, `select`, `vary`, `relevel`, `GeneTree`, `neighbourhood`, the cleanse and literal mutations); the host `init`, the old `Population` with a constants array per gene, `DepthBudget`, `TypeWalk` and the untyped `TypedSampling`; `TypedCodes::fallback_op`, `fallback_k` and `typed_kingdom`; `KingdomTypes::fallbacks` and `literals`, `SymbolTable::fallback_leaf`, `decode_gene_masked`, `LiteralCodes`, `Rates::lit_point`. `decode_gene` (host) now refuses a token of the wrong type, and a literal placeholder decodes like `?_T`, an index into `C_T`. Every host rewrite of an expressed gene (cleanse, snap, fold, the beam, regex simplify) goes through fuller's `express` and `write_back` (`typed_write_back.rs`: `GeneCodec`), and a rewrite that does not fit is a named defect. The subexpression search now breeds on the typed device. The fold operator on by default in SR is the near-constant-blob collapse, on the codec; fuller's ADF fold (shared subtrees) is a different operator and is not on by default.
