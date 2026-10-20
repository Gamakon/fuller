-- MultiTypedGEP.md revision 4, section 3, verbatim: the views that derive the
-- bounds, then the recursive expression query, which binds the chromosome id as
-- its one parameter. The test splits the file at the blank line that precedes
-- the query.

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
