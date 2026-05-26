# Evolution as SQL
## Tournament Selection · Crossover · Mutation · v0.1 · 2026-03-11

> All evolutionary operators are **read-only SELECT statements** that produce
> the next generation as new rows. The geneframe is append-only. No existing
> rows are ever modified. History is the fossil record.

---

## 1. The Append-Only Principle

Classical GEP evolves chromosomes in-place: objects are mutated, lists are
spliced. nucleotable never modifies existing rows.

Each generation step is:
```sql
INSERT INTO geneframe
SELECT [next generation] FROM [current generation via CTEs];
```

The previous generation remains intact. `generation_id` increments. The full
evolutionary history is queryable at any time.

---

## 2. Head and Tail Selection

`gene_seq_id` is the positional column. It is the vertical axis of the
chromosome. All evolutionary operations are predicates on this column.

```sql
-- Select the head of a chromosome (HeadLength = 8)
SELECT * FROM geneframe
WHERE chromosome_id = :chrom
  AND gene_seq_id <= 8;       -- or: AND ishead = TRUE

-- Select the tail
SELECT * FROM geneframe
WHERE chromosome_id = :chrom
  AND gene_seq_id > 8;        -- or: AND ishead = FALSE

-- Select genes up to a crossover point
SELECT * FROM geneframe
WHERE chromosome_id = :chrom
  AND gene_seq_id <= :crossover_point;

-- Select genes after a crossover point
SELECT * FROM geneframe
WHERE chromosome_id = :chrom
  AND gene_seq_id > :crossover_point;
```

The head/tail boundary is a fixed crossover point (`HeadLength`).
The genetic crossover point is a random one. Same operation, different value.

---

## 3. Fitness Evaluation

Fitness is a UDF that takes a chromosome and returns a scalar score.
It is registered as a SQL function (via DuckDB UDF API or WASM extension).

```sql
-- Evaluate fitness for all individuals in the current generation
SELECT
    individual_id,
    chromosome_id,
    fitness_udf(chromosome_id)  AS score
FROM geneframe
WHERE population_id  = :pop
  AND generation_id  = :gen
GROUP BY individual_id, chromosome_id;
```

`fitness_udf` decodes the Karva chromosome (BFS tree reconstruction),
evaluates it against training data, and returns a scalar fitness score.
The implementation is kingdom-specific:
- Symbolic Regression: mean squared error against target values
- NLP-English: semantic similarity or information density score
- SQL: query execution result correctness
- BotjiKingdom: address navigation accuracy

---

## 4. Tournament Selection

Tournament selection assigns individuals to random groups and picks the
winner from each group. No sorting of the full population needed.

```sql
WITH

fitness AS (
    SELECT
        individual_id,
        chromosome_id,
        fitness_udf(chromosome_id)                       AS score
    FROM geneframe
    WHERE population_id = :pop AND generation_id = :gen
),

tournaments AS (
    SELECT
        individual_id,
        chromosome_id,
        score,
        CAST(RANDOM() * :tournament_size AS INTEGER)     AS tournament_id
    FROM fitness
),

winners AS (
    SELECT individual_id, chromosome_id
    FROM (
        SELECT
            individual_id,
            chromosome_id,
            RANK() OVER (
                PARTITION BY tournament_id
                ORDER BY score DESC
            )                                            AS fitness_rank
        FROM tournaments
    )
    WHERE fitness_rank = 1
)

SELECT * FROM winners;
```

`RANDOM() * :tournament_size` is the entire tournament mechanism.
Partition by the random bucket. Take the top-ranked individual per bucket.
No sorting, no loops.

---

## 5. Crossover (One-Point Recombination)

Crossover generates offspring by splitting two parent chromosomes at a random
point and swapping the tail fragments.

**Key insight:** because chromosomes are vertical (one row per gene), crossover
is a `<= / >` split on `gene_seq_id` followed by a shuffle of individual IDs.
No list splicing. No object manipulation. Pure set operations.

```sql
WITH

-- Step 1: One random crossover point per individual (between 1 and chr_len)
crossover_points AS (
    SELECT DISTINCT
        chromosome_id,
        individual_id,
        CAST(RANDOM() * :chr_len AS INTEGER) + 1   AS cp
    FROM geneframe
    WHERE population_id = :pop AND generation_id = :gen
),

-- Step 2: Before fragment — stays with its original individual
fragment_before AS (
    SELECT g.*
    FROM geneframe g
    JOIN crossover_points cp USING (chromosome_id)
    WHERE g.gene_seq_id <= cp.cp
),

-- Step 3: After fragment — will be shuffled to a different individual
fragment_after AS (
    SELECT g.*
    FROM geneframe g
    JOIN crossover_points cp USING (chromosome_id)
    WHERE g.gene_seq_id > cp.cp
),

-- Step 4: Shuffle individual IDs on the after-fragments
-- Rank both donor and recipient pools randomly, join on rank → swap
donor_ranked AS (
    SELECT individual_id,
           ROW_NUMBER() OVER (ORDER BY RANDOM())   AS rn
    FROM (SELECT DISTINCT individual_id FROM fragment_after)
),
recipient_ranked AS (
    SELECT individual_id,
           ROW_NUMBER() OVER (ORDER BY RANDOM())   AS rn
    FROM (SELECT DISTINCT individual_id FROM fragment_before)
),
shuffled_after AS (
    SELECT
        fa.* EXCLUDE (individual_id),
        r.individual_id                            AS individual_id
    FROM fragment_after fa
    JOIN donor_ranked    d  ON fa.individual_id = d.individual_id
    JOIN recipient_ranked r ON d.rn = r.rn
),

-- Step 5: Recombine — each individual now has its own head + someone else's tail
offspring AS (
    SELECT * FROM fragment_before
    UNION ALL
    SELECT * FROM shuffled_after
)

-- Step 6: Write next generation
INSERT INTO geneframe
SELECT
    :pop            AS population_id,
    :next_gen       AS generation_id,
    o.*
FROM offspring o;
```

Each individual's `gene_seq_id <= crossover_point` rows are preserved.
Each individual's `gene_seq_id > crossover_point` rows are replaced by
another individual's tail fragment (randomly assigned via ROW_NUMBER shuffle).

The result: offspring chromosomes with mixed genetic material from the
winner population. One INSERT. No loops.

---

## 6. Mutation

Mutation randomly replaces gene symbols with probability `μ` (mutation rate).
Applied to the winner population, generating variant offspring.

```sql
WITH winners AS ( ... )  -- from tournament selection above

INSERT INTO geneframe
SELECT
    :pop                                            AS population_id,
    :next_gen                                       AS generation_id,
    w.individual_id,
    w.chromosome_id,
    g.gene_seq_id,
    g.kingdom_id,
    g.ishead,
    CASE
        WHEN RANDOM() < :mu AND g.ishead = TRUE
        THEN random_symbol(:kingdom_id)             -- random function from head
        WHEN RANDOM() < :mu AND g.ishead = FALSE
        THEN random_terminal(:kingdom_id)           -- random terminal from tail
        ELSE g.symbol
    END                                             AS symbol,
    g.botji_addr,
    g.raw_text
FROM winners w
JOIN geneframe g USING (chromosome_id)
WHERE g.population_id = :pop AND g.generation_id = :gen;
```

`random_symbol` and `random_terminal` are UDFs that draw a random symbol ID
from the appropriate kingdom's symbols or terminals table. They maintain the
head/tail type constraint: the head can only receive functions; the tail can
only receive terminals.

---

## 7. The Full Generation Step

One complete evolutionary cycle — selection, crossover, mutation — as a
single chained CTE:

```sql
WITH

-- Fitness
fitness AS ( ... ),
tournaments AS ( ... ),
winners AS ( ... ),

-- Crossover
crossover_points AS ( ... ),
fragment_before AS ( ... ),
fragment_after AS ( ... ),
shuffled_after AS ( ... ),
post_crossover AS (
    SELECT * FROM fragment_before
    UNION ALL
    SELECT * FROM shuffled_after
),

-- Mutation
next_generation AS (
    SELECT
        :pop AS population_id,
        :next_gen AS generation_id,
        individual_id,
        chromosome_id,
        gene_seq_id,
        kingdom_id,
        ishead,
        CASE
            WHEN RANDOM() < :mu AND ishead = TRUE  THEN random_symbol(:kingdom)
            WHEN RANDOM() < :mu AND ishead = FALSE THEN random_terminal(:kingdom)
            ELSE symbol
        END AS symbol
    FROM post_crossover
)

INSERT INTO geneframe SELECT * FROM next_generation;
```

One statement. One INSERT. The entire generation is produced by SQL.

---

## 8. Multi-Typed TailLength — The Hard Part

Classical GEP TailLength is straightforward:
```
TailLength = HeadLength × (MaxArity - 1) + 1
MaxArity   = max total input arity across all symbols
```

In multi-typed GEP (NLP-English, BotjiKingdom), the tail must satisfy a
stricter constraint: it needs enough terminals **of the right types** to fill
any possible combination of typed argument slots across all head functions.

The classical formula still gives the correct *length* — MaxArity is computed
as the sum of all input arities for the most complex function:

```sql
-- NLP-English MaxArity
SELECT MAX(
    in_PERSON + in_ORG + in_GPE + in_LOC + in_NORP + in_FAC +
    in_EVENT  + in_PRODUCT + in_DATE + in_TIME + in_MONEY +
    in_QUANTITY + in_CARDINAL + in_PERCENT + in_NP + in_AP + in_CLAUSE + in_VERB
) AS max_arity
FROM symbols WHERE kingdom = 'NLP-English';
```

But the *composition* of the tail changes. The tail is not a homogeneous pool
of interchangeable terminals — it is a typed sequence where each position must
be fillable by the type demanded at that slot in the expression tree's BFS
expansion.

**Implications for the tail generator:**
- The random tail must draw terminals proportional to the type demands of the
  head functions (not uniformly at random)
- A head with `strike` (in_ORG=1, in_GPE=1, in_LOC=1, in_NORP=1, in_FAC=1)
  requires five typed terminal slots — one of each type
- Padding with `NULL` terminals (symbol=-999, out_NP=1) fills positions where
  no typed terminal is available

**This is not `SELECT *` work.** The typed tail generator requires:
1. BFS expansion of the head to determine which typed slots need filling
2. Type-aware terminal sampling (draw a GPE terminal for a GPE slot)
3. Tail padding to fixed chromosome length
4. Type-compatible crossover validation at splice points

This is a solved problem in typed GEP theory — but the SQL implementation
requires actual thinking about type propagation through the expression tree.
It is documented here as a known hard requirement, not a solved one.

---

## 9. Why This Works

The vertical table representation is what makes this possible.

In the horizontal (list) representation:
- Crossover requires slicing lists and concatenating: `a[:k] + b[k:]`
- That operation is not expressible in SQL
- You need a loop, a cursor, or an ORM

In the vertical (table) representation:
- Crossover is `WHERE gene_seq_id <= k` and `WHERE gene_seq_id > k`
- Those are standard SQL predicates
- The shuffle is `ROW_NUMBER() OVER (ORDER BY RANDOM())`
- The whole thing is a set operation

The representation determines the operations that are natural.
Rotating the chromosome 90 degrees rotated the available toolset with it.
SQL, DuckDB, Parquet, WASM — all of these were already built for this shape.
