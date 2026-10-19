# Multi-typed GEP: one head, one tail per type

## Context

**Objective.** Multi-typed gene expression programming in which every expressed node has the type required by its parent's argument slot, for every gene and every permitted variation. There is no repair, rejection, substitution, or fallback symbol: validity is a property of the layout.

**Approach.** A gene has one shared head plus one tail for each output type. A slot requiring `T` reads the next unused member of `T`'s reading frame: head rows whose symbol outputs `T`, in head order, followed by the `T` tail. A `T` tail contains terminals outputting `T` only. Thus a `T` slot cannot select a symbol of another type.

This is a **typed extension of Karva**, not classical Karva when a kingdom has more than one type. Classical Karva consumes one ordered head prefix globally. Here head rows are consumed in per-type frames, so a level-order walk can visit head positions out of physical head order. With head `[left:S(S,I), i:I, s:S]` and root `S`, the walk reads positions `0, 2, 1` and produces `left(s, i)`. It retains neutral material, but loses Karva's global head-prefix property and some positional locality: changing a head row's output type moves it between frames and can shift later selections in both frames. This is an accepted cost of Andrew's one-head, one-tail-per-type design; do not replace it with a single globally consumed head stream.

**Status.** Design only. The current shared-tail/fallback projections are incompatible with this design and are to be removed during implementation.

## 1. Kingdom contract

`symbol_outputs(kingdom_id, symbol, output_type)` is a unified relation over functions and terminals. Every symbol has exactly one output type. A polymorphic spelling is represented by one distinct symbol row per output type.

Argument order is data, not an implication of type counts:

```
argument_signature(kingdom_id, symbol, arg_index, required_type)
PRIMARY KEY (kingdom_id, symbol, arg_index)
```

`arg_index` is contiguous from zero for each function. A terminal has no signature rows. The derived count `in_T` is the number of signature rows requiring `T`; it is not used to order a call. Thus `left:S(S,I)` has `(left,0,S)` then `(left,1,I)`.

The kingdom-load check must establish all of the following.

- Exactly one `symbol_outputs` row exists per symbol; zero-output and multi-output rows are forbidden.
- Each signature belongs to a function, has one contiguous ordered index sequence, and names a kingdom type. A terminal has zero arguments.
- Each root type and each type occurring in `required_type` has at least one terminal.
- `a_T = max_s count(arguments of s whose required_type = T)`, with zero if none; and `n_max = max_s count(arguments of s)` over the same function rows (zero if the kingdom has no functions).

The ordered signature relation, one output per row, and these definitions are all part of the kingdom check.

## 2. Gene layout and bound

Each row has an unambiguous domain-local key:

```
(chromosome_id, gene_id, domain, domain_type, domain_seq_id)
```

`domain` is `head`, `tail`, or `dc`; `domain_type` is NULL for a head and is the type of a tail or Dc domain; `domain_seq_id` starts at zero within that domain. `gene_id` identifies a fixed gene boundary, not a breadth-first position. Head rows store `symbol`; tail rows store a terminal `symbol`; Dc rows store `dc_index`, never a symbol.

For head length `h`, root type `R`, and type `T`, use:

```
t_T = min(h * a_T + [T = R], h * (n_max - 1) + 1)
```

Do not create a tail for a non-root type with `a_T = 0`. At most `h` expressed head functions open slots, hence `D_T <= h*a_T + [T=R]`; and there are at most `h*(n_max-1)+1` leaves. A tail entry is a leaf used only for a same-type demand, so the minimum is safe.

With one type, every head row is in its one frame, `a_T = n_max`, and the frame is the physical head followed by its tail. Therefore `t_T = h*(n_max-1)+1`, and the deterministic walk below is exactly classical Karva decoding: by induction each next open slot consumes the next physical head row until the head ends, then the next tail row, and appends the same ordered arguments.

## 3. Deterministic expression

Start a FIFO queue with one slot `(R, depth 0)`. Keep `k_T`, initially zero, for each type. On each dequeue of a `T` slot, select frame row `T[k_T]`, increment `k_T`, and append that symbol's ordered argument-signature slots to the queue. Stop at an empty queue. A tail cannot be exhausted by §2, and selected output type equals the slot type by the frame construction.

Unused rows are neutral. Unlike classical Karva, these need not be one final physical suffix of the head, because each type has its own head frame.

### Executable DuckDB expression query

The following names the required relations and implements the complete FIFO level-order walk. It uses `LIST` state so that `used` supplies the per-type rank before the current slot is added. `step` is the level-order node number; `parent_step` and `arg_index` retain the ordered tree edges.

```sql
CREATE OR REPLACE VIEW symbol_outputs AS
SELECT kingdom_id, symbol, output_type FROM functions
UNION ALL
SELECT kingdom_id, symbol, output_type FROM terminals;

WITH RECURSIVE
frame_source AS (
  SELECT g.chromosome_id, g.gene_id, g.kingdom_id, o.output_type, 0 AS tail_part,
         g.domain_seq_id, g.symbol
  FROM geneframe AS g
  JOIN symbol_outputs AS o
    ON o.kingdom_id = g.kingdom_id AND o.symbol = g.symbol
  WHERE g.domain = 'head'
  UNION ALL
  SELECT g.chromosome_id, g.gene_id, g.kingdom_id, g.domain_type AS output_type, 1 AS tail_part,
         g.domain_seq_id, g.symbol
  FROM geneframe AS g
  JOIN symbol_outputs AS o
    ON o.kingdom_id = g.kingdom_id AND o.symbol = g.symbol
   AND o.output_type = g.domain_type
  WHERE g.domain = 'tail'
),
frames AS (
  SELECT *, row_number() OVER (
    PARTITION BY chromosome_id, gene_id, kingdom_id, output_type
    ORDER BY tail_part, domain_seq_id
  ) - 1 AS frame_k
  FROM frame_source
),
walk(chromosome_id, gene_id, kingdom_id, step, pending, used,
     node_type, frame_k, symbol, depth, parent_step, arg_index) AS (
  SELECT ge.chromosome_id, ge.gene_id, ge.kingdom_id, 0,
         list_value(struct_pack(required_type := ge.root_type,
                                depth := 0::BIGINT,
                                parent_step := NULL::BIGINT,
                                arg_index := NULL::BIGINT)),
         []::VARCHAR[],
         NULL::VARCHAR, NULL::BIGINT, NULL::VARCHAR,
         NULL::BIGINT, NULL::BIGINT, NULL::BIGINT
  FROM genes AS ge
  WHERE ge.chromosome_id = ?  -- bind the selected chromosome_id

  UNION ALL

  SELECT w.chromosome_id, w.gene_id, w.kingdom_id, w.step + 1,
         list_concat(list_slice(w.pending, 2, len(w.pending)), coalesce(kids.slots, list_value())),
         list_append(w.used, q.slot.required_type),
         q.slot.required_type,
         len(list_filter(w.used, x -> x = q.slot.required_type)),
         f.symbol, q.slot.depth, q.slot.parent_step, q.slot.arg_index
  FROM walk AS w
  CROSS JOIN LATERAL (SELECT list_extract(w.pending, 1) AS slot) AS q
  JOIN frames AS f
   ON f.chromosome_id = w.chromosome_id
   AND f.gene_id = w.gene_id
   AND f.kingdom_id = w.kingdom_id
   AND f.output_type = q.slot.required_type
   AND f.frame_k = len(list_filter(w.used, x -> x = q.slot.required_type))
  LEFT JOIN LATERAL (
    SELECT list(struct_pack(required_type := a.required_type,
                            depth := q.slot.depth + 1,
                            parent_step := w.step + 1,
                            arg_index := a.arg_index)
                ORDER BY a.arg_index) AS slots
    FROM argument_signature AS a
    WHERE a.kingdom_id = w.kingdom_id AND a.symbol = f.symbol
  ) AS kids ON TRUE
  WHERE len(w.pending) > 0
)
SELECT step, node_type, frame_k, symbol, depth, parent_step, arg_index
FROM walk
WHERE step > 0
ORDER BY gene_id, step;
```

The `frames` partition is deliberately `(chromosome_id, gene_id, output_type)`, and ordering is by `domain_seq_id` inside head or the matching typed tail. It cannot merge two genes or two same-type tails. The decoder must report a kingdom/layout defect if the join finds no frame row; it must not repair it.

### Two-type example (`left:S(S,I)`)

For one gene with root `S`, load head rows `(0,left)`, `(1,i)`, `(2,s)` and enough typed tails (`S` tail symbols `s`, `I` tail symbols `i`). The function signature is `(left,0,S)` and `(left,1,I)`; `s` and `i` are terminals. The query above produces:

| step | node_type | frame_k | symbol | depth | parent_step | arg_index |
|---:|---|---:|---|---:|---:|---:|
| 1 | S | 0 | left | 0 | NULL | NULL |
| 2 | S | 1 | s | 1 | 1 | 0 |
| 3 | I | 0 | i | 1 | 1 | 1 |

The expressed tree is `left(s,i)`, with physical head positions `0,2,1`. This example is also a regression test for the non-Karva multi-type ordering.

## 4. Constants (GEP-RNC, per type)

A constant-capable type `T` has placeholder terminal `?_T` and one `Dc_T` domain of length `t_T`. Keep a separate `q_T`, initially zero, for every type. Only when an **expressed** `?_T` is visited does it read `C_T[Dc_T[q_T]]` and increment `q_T`; unexpressed placeholders and placeholders of other types do nothing. Thus a sole expressed root `?_I` uses `Dc_I[0]`, even if later head placeholders exist.

`C_T` is a kingdom/type constant array, append-only for the life of a population epoch: a new constant appends a value; existing indices are immutable and never renumbered. Dc mutation, recombination, inversion, and transposition are local to the matching `(gene_id, T, dc)` domain and choose only valid indices of that `C_T`. This preserves every existing reference while the array evolves.

## 5. Variation: domain alphabets are invariant

Every operation preserves fixed domain lengths and its target domain's alphabet.

| operator | permitted source and target |
|---|---|
| mutation | head: any kingdom symbol; tail-T: a terminal outputting T; Dc-T: a valid index into `C_T` |
| inversion | within one exact `(gene, domain, domain_type)` domain |
| IS transposition | copy a sequence from a head or tail-T domain into the head only; head admits all symbols; source Dc is forbidden |
| RIS transposition | copy a head sequence beginning with a function into the head only; Dc and tail sources are forbidden |
| tail transposition | copy/reorder only within the same tail-T domain |
| Dc transposition | copy/reorder only within the same Dc-T domain; Dc never enters head or tail |
| one-/two-point and gene recombination | homologous coordinates only: same kingdom, layout id, gene id, domain, domain type, and local positions |

IS/RIS keep the fixed head length by truncating the copied-in head result at `h`; no non-symbol can enter it. Homologous gene recombination exchanges whole matching genes or matching domain segments, never an unaligned global position. A changed head output type may redistribute future frame use and is therefore less local than classical Karva, but remains type-correct.

## 6. Multigenic chromosomes and homeotic genes

A chromosome has an ordered `genes` relation with explicit `gene_id`, `kingdom_id`, root type, and layout id; each gene owns precisely its head, tails, and Dc domains. Its phenotype is defined by a typed linker symbol with an ordered signature over the gene roots (for two `R` genes, for example, `link:R(R,R)`). A chromosome is not merely a bag of evaluated genes.

A homeotic definition stores a stable immutable definition id and its declared root type. A homeotic-reference terminal outputs that root type and names the stable id, never `gene_seq_id`. Definitions visible to a chromosome are immutable during its mutation and recombination. They are assigned levels, and a definition at level `L` may reference only definitions at lower levels. The strict decrease proves acyclicity; references that are missing, out of scope, or do not match the declared root type are kingdom/layout errors, not fallback opportunities.

## 7. Implementation and tests

Remove all fallback/substitution paths from the host and GPU decoders; an unread or unavailable required value is a reported kingdom/layout defect. Implement per-type frames and cursors in both decoders, and use the same ordered-signature table.

1. **Kingdom check.** Assert the contract in §1, including ordered signatures and terminal coverage.
2. **Closure.** For SR, regex, and WGSL interfaces: 100,000 random genes and 100,000 children per operator. Every expressed node equals its slot type; no decoder substitutes.
3. **Bounds and Dc.** Record maximum tail and `q_T` indices; each is below `t_T`. Exercise unexpressed and interleaved placeholders and verify only expressed `?_T` advances `q_T`.
4. **SQL.** Run the query in §3 in `duckdb :memory:` with the two-type fixture and assert the table shown there, including frame partitions across two genes and two chromosomes.
5. **Karva equality.** For each one-type SR gene, typed decoding equals classical Karva exactly.
6. **Determinism and round trip.** Host and GPU produce identical trees; WGSL seeds re-encode and express to the same tree. Include multigenic linker and lower-level-only homeotic cases.

## 8. Build order

1. Add the kingdom contract, domain-local keys, layout, host decoder, and SQL fixture.
2. Add GPU per-type cursors and remove fallback bindings; establish host/GPU parity.
3. Implement domain-preserving operators and closure tests.
4. Move SR and regex, then WGSL; re-encode saved genomes and run round-trip tests.

## Revisions (2026-10-19, after review)

- Made the representation's non-classical multi-type head order and locality cost explicit.
- Added ordered argument signatures, full kingdom checks, domain-local keys, and executable frame/walk SQL with a two-type result.
- Restricted transposition and recombination to alphabet-preserving domains; Dc never enters head.
- Specified gene boundaries, typed linking, homeotic identity/levels, and expressed-only per-type Dc counters with append-only constant arrays.
