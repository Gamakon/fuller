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
