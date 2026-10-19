# Multi-typed GEP: a head and tail for every output type

## Context

**Objective.** Multi-typed gene expression programming in which every expressed node has the type required by its parent's argument slot, for every gene and every permitted variation. Validity is a property of the layout: there is no repair, rejection, substitution, or fallback symbol.

**Approach.** A gene has a separate `head_T` and `tail_T` for every output type `T`. A slot requiring `T` reads the next unused member of `T`'s frame: `head_T`, then `tail_T`. `head_T` contains only symbols whose output is `T`; `tail_T` contains only terminals whose output is `T`. This is the design: it preserves type correctness *and* makes mutation and crossover local to the output-type domain.

**Status.** Design only. The current shared-tail/fallback projections are incompatible with it and are to be removed during implementation.

## 1. Kingdom contract

`symbol_outputs(kingdom_id, symbol, output_type)` is a unified relation over functions and terminals. Every symbol has exactly one output type; a polymorphic spelling has one distinct row per output type.

```
argument_signature(kingdom_id, symbol, arg_index, required_type)
PRIMARY KEY (kingdom_id, symbol, arg_index)
```

`arg_index` is contiguous from zero for each function. A terminal has no signature rows. Thus `left:S(S,I)` has `(left,0,S)` and `(left,1,I)`: argument order is data, not an implication of type counts.

The kingdom check establishes exactly one output row per symbol; ordered, contiguous signatures on functions only; and at least one terminal for every root or required type. Define `a_T = max_s count(arguments of s whose required_type = T)` (zero if none), and `n_max = max_s arity(s)` (zero if there are no functions).

## 2. Gene layout, heads, and bounds

Each row has the domain-local key:

```
(chromosome_id, gene_id, domain, domain_type, domain_seq_id)
```

`domain` is `head`, `tail`, or `dc`; `domain_type` is the output type for every head, tail, and Dc domain; and `domain_seq_id` starts at zero within that domain. A head row stores a symbol outputting `domain_type`; a tail row stores a terminal outputting it; a Dc row stores `dc_index`, never a symbol. `gene_id` is a fixed gene boundary, not a breadth-first position.

For a gene layout choose fixed per-type head lengths `h_T`; they are layout parameters, not evolved values. Give every type reachable from the root a positive `h_T` (an unreachable type may have zero), normally allocating a fixed total head budget across types according to the kingdom's expected use. Equal `h_T` is the neutral default; a kingdom may deliberately give a specialist type more capacity. Define `H = sum_T h_T`. Fixed domain lengths make homologous crossover meaningful.

At most `H` expressed head functions can open slots. Therefore a demand of type `T` is bounded both by `H * a_T + [T = R]` and by the total number of leaves, `H * (n_max - 1) + 1`. Use:

```
t_T = min(H * a_T + [T = R], H * (n_max - 1) + 1).
```

Do not create a non-root `tail_T` when `a_T = 0`; a root tail exists even when its head is empty. This bound is safe even though functions come from all `head_U`: each of the at most `H` functions contributes at most `a_T` `T` slots.

With one type, `H = h_T`, its sole head is the physical Karva head, `a_T = n_max`, and `t_T = h_T * (n_max - 1) + 1`. The decoder below is exactly classical Karva. With several types this is better than a shared head: changing a symbol cannot move it between type frames or shift a different domain's read cursor. It restores within-domain Karva locality and homologous crossover, at the cost of choosing a fixed head-budget allocation among types.

## 3. Deterministic expression

Start FIFO `pending` with `(R, depth 0)`. Keep `k_T = 0` for every type. On dequeue of a `T` slot, select row `T[k_T]` from `head_T` followed by `tail_T`, increment `k_T`, and append the selected symbol's ordered-signature slots. Stop at an empty queue. The bound prevents tail exhaustion and frame membership proves the selected symbol has type `T`.

### Executable DuckDB expression query

The query names the required relations and implements the complete FIFO walk. `used` supplies the per-type rank before the current slot is added; `parent_step` and `arg_index` retain ordered edges.

```sql
CREATE OR REPLACE VIEW symbol_outputs AS
SELECT kingdom_id, symbol, output_type FROM functions
UNION ALL
SELECT kingdom_id, symbol, output_type FROM terminals;

WITH RECURSIVE
frame_source AS (
  SELECT g.chromosome_id, g.gene_id, g.kingdom_id, g.domain_type AS output_type,
         0 AS tail_part, g.domain_seq_id, g.symbol
  FROM geneframe AS g JOIN symbol_outputs AS o
    ON o.kingdom_id = g.kingdom_id AND o.symbol = g.symbol
   AND o.output_type = g.domain_type
  WHERE g.domain = 'head'
  UNION ALL
  SELECT g.chromosome_id, g.gene_id, g.kingdom_id, g.domain_type AS output_type,
         1 AS tail_part, g.domain_seq_id, g.symbol
  FROM geneframe AS g JOIN symbol_outputs AS o
    ON o.kingdom_id = g.kingdom_id AND o.symbol = g.symbol
   AND o.output_type = g.domain_type
  WHERE g.domain = 'tail'
),
frames AS (
  SELECT *, row_number() OVER (
    PARTITION BY chromosome_id, gene_id, kingdom_id, output_type
    ORDER BY tail_part, domain_seq_id
  ) - 1 AS frame_k FROM frame_source
),
walk(chromosome_id, gene_id, kingdom_id, step, pending, used,
     node_type, frame_k, symbol, depth, parent_step, arg_index) AS (
  SELECT ge.chromosome_id, ge.gene_id, ge.kingdom_id, 0,
         list_value(struct_pack(required_type := ge.root_type, depth := 0::BIGINT,
                                parent_step := NULL::BIGINT, arg_index := NULL::BIGINT)),
         []::VARCHAR[], NULL::VARCHAR, NULL::BIGINT, NULL::VARCHAR,
         NULL::BIGINT, NULL::BIGINT, NULL::BIGINT
  FROM genes AS ge WHERE ge.chromosome_id = ?
  UNION ALL
  SELECT w.chromosome_id, w.gene_id, w.kingdom_id, w.step + 1,
         list_concat(list_slice(w.pending, 2, len(w.pending)), coalesce(kids.slots, list_value())),
         list_append(w.used, q.slot.required_type), q.slot.required_type,
         len(list_filter(w.used, x -> x = q.slot.required_type)), f.symbol,
         q.slot.depth, q.slot.parent_step, q.slot.arg_index
  FROM walk AS w CROSS JOIN LATERAL (SELECT list_extract(w.pending, 1) AS slot) AS q
  JOIN frames AS f ON f.chromosome_id = w.chromosome_id AND f.gene_id = w.gene_id
    AND f.kingdom_id = w.kingdom_id AND f.output_type = q.slot.required_type
    AND f.frame_k = len(list_filter(w.used, x -> x = q.slot.required_type))
  LEFT JOIN LATERAL (
    SELECT list(struct_pack(required_type := a.required_type, depth := q.slot.depth + 1,
                            parent_step := w.step + 1, arg_index := a.arg_index) ORDER BY a.arg_index) AS slots
    FROM argument_signature AS a WHERE a.kingdom_id = w.kingdom_id AND a.symbol = f.symbol
  ) AS kids ON TRUE WHERE len(w.pending) > 0
)
SELECT step, node_type, frame_k, symbol, depth, parent_step, arg_index
FROM walk WHERE step > 0 ORDER BY gene_id, step;
```

The decoder reports a kingdom/layout defect when a frame row is missing; it never repairs. The partition deliberately includes chromosome, gene, and type.

For `left:S(S,I)`, with `head_S=(left,s)`, `head_I=(i)`, root `S`, and typed terminal tails, the result is `left(s,i)`: steps are `(S,0,left)`, `(S,1,s)`, `(I,0,i)`. The physical rows read are in their own domains, not an inter-type shared stream.

## 4. Constants (GEP-RNC, per type)

A constant-capable type `T` has placeholder terminal `?_T` and `Dc_T` of length `t_T`. Only an **expressed** placeholder consumes the next `Dc_T` entry and advances `q_T`. `C_T` is append-only for a population epoch; existing indices are immutable. Dc mutation, recombination, inversion, and transposition stay in their matching `(gene_id, T, dc)` domain and select valid `C_T` indices.

## 5. Variation: domains are invariant

Mutation selects any kingdom symbol outputting `T` in `head_T`, a terminal outputting `T` in `tail_T`, or a valid index in `Dc_T`. Inversion and tail/Dc transposition stay within one exact `(gene, domain, type)` domain. IS and RIS may copy a head segment into the *same* `head_T` only (RIS starts at a function); no tail or Dc material enters a head. One-/two-point and gene recombination exchange homologous coordinates only: same kingdom, layout id, gene id, domain, output type, and local positions. All operations preserve domain lengths and alphabets. This is why the per-type-head design restores locality: an edit in `head_T` affects only `T`'s frame.

## 6. Typed linkers, ADFs, and multicellular programs

A chromosome's normal genes are combined by a typed linker. It may be a fixed linker function whose ordered signature consumes the normal-gene root types, or Ferreira's evolved **homeotic gene**. The homeotic gene has its own `head_T` and `tail_T` layout from §2. Its heads contain only typed linking functions and genic terminals outputting `T`; its tails contain genic terminals only. For each type `T`, there is one distinct genic terminal for each linked normal gene whose root is `T` (`g_T` choices, with repetition permitted in the homeotic gene). Thus its terminal alphabet is typed and its inventory is counted by `g_T`, not by an untyped gene position. Layout validation requires `g_T > 0` for every type that the linker can demand.

The homeotic gene's declared root is the chromosome phenotype type; expression invokes each selected genic terminal's named normal gene. It consequently evolves both which normal genes participate and how typed linking functions compose them, while a fixed linker is the non-evolving alternative. Several homeotic genes mean several main programs (a multicellular system): each has a stable main-program id, declared phenotype type, and linked normal-gene set; evaluate each independently, retaining their ordered result tuple.

**Ferreira names.** Population-wide definitions found by share-equivalence are **ADFs**: ordinary, typed normal genes, not homeotic genes. Fold turns each selected shared DAG node into an immutable ADF gene with a stable `adf_id`, root type, and level; it rewrites the remainder into the homeotic gene(s), whose typed genic terminals call those ADFs (or chromosome-local normal genes). An ADF reference is therefore a typed genic terminal: it carries `adf_id`, outputs that ADF's declared root type, and is valid only in that type's head or tail. An ADF at level `L` may call only strictly lower-level ADFs. Stable ids, root types, immutability, and strict level decrease give persistent references and acyclicity.

Required code/document rename (not performed by this document-only change): in `docs/PLAN_population_dag.md`, replace “homeotic chromosome/row/tail/definition” when it denotes a population-wide shared definition with “ADF gene/row/set/definition”; retain “homeotic gene” only for the rewritten chromosome linker. In `src/population_dag.rs`, rename `is_homeotic`, `homeotic`, `homeotic_new`, related fields/comments/tests, and memo-selection terminology to `is_adf`, `adfs`, `adf_new`, and ADF. In `src/homeotic.rs`, rename the share-finder fold product, `FoldedChromosome` head/tail documentation, href vocabulary, and tests from “homeotic tail/slot” to typed ADF genes/`adf_id`; reserve `homeotic` for the new linker implementation. The equivalent phylu `GENEREF`, export, checkpoint, arena, and level-schedule names must make the same distinction.

## 7. Implementation and tests

1. Kingdom/layout checks: ordered signatures, terminal coverage, `h_T`, the `H` tail bound, and per-domain alphabets.
2. Host and GPU closure: 100,000 random genes and children per operator for SR, regex, and WGSL; every node matches its slot type and neither decoder substitutes.
3. Bounds/constants: record maximum `k_T` and `q_T`, both below `h_T+t_T` and `t_T` respectively; include expressed-only, interleaved placeholders.
4. SQL: run §3 in `duckdb :memory:` with the two-type fixture, plus two genes and two chromosomes, asserting partitions and the stated tree.
5. Locality/Karva: edits in `head_T` do not alter another frame; same-type crossover is homologous; one-type genes decode byte-for-byte as classical Karva.
6. Linkers/ADFs: fixed and evolved linkers, genic-terminal type/count checks, multiple homeotic main programs, stable ADF ids, immutable typed references, and lower-level-only ADF calls. Verify fold → typed ADFs + rewritten homeotic gene(s) → unfold is value-preserving.

## 8. Build order

1. Add kingdom checks, typed domain keys, per-type heads/tails, host decoder, and DuckDB fixture.
2. Add host operators and the locality, closure, bound, and one-type Karva tests.
3. Add GPU per-type cursors and parity; remove all fallback bindings.
4. Implement share-equivalence output as ADF genes plus typed homeotic linkers, stable ids/levels, and the ADF rename across fuller and phylu.
5. Move SR and regex, then WGSL; re-encode saved genomes and run round trips.

## Revisions (2026-10-20)

- Replaced the shared head with a typed head and tail for every output type; made it the chosen design because it restores per-domain locality and homologous crossover.
- Derived the multi-head tail bound using `H = sum_T h_T`, specified fixed head-length selection, and retained exact one-type Karva reduction.
- Adopted Ferreira terminology: shared population definitions are ADF normal genes; the rewritten linker remainder is the homeotic gene.
- Specified typed genic terminals, evolved/fixed linkers, multicellular multiple-homeotic-program systems, and the share-equivalence-to-ADF/homeotic transformation.
