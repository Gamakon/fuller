# GeneFrame Data Model
## Entity-Relation Design · v0.1 · 2026-03-11

> Classical GEP defines the Karva language and its rules (Ferreira, 2006).
> nucleotable's contribution is to **turn genes into rows** — holding genomes
> and populations in dataframes, where each gene carries its attributes as
> typed arity columns, many-hot encoded.
>
> This transforms GEP from an in-memory object graph into a relational system
> that can be queried, evolved, and exchanged as SQL, Parquet, or Arrow.

---

## 1. Entity-Relation Overview

```
┌─────────────┐        ┌──────────────────────────────────────┐        ┌────────────────┐
│   KINGDOM   │        │              GENEFRAME               │        │  SYMBOL TABLE  │
│─────────────│        │           (transaction table)        │        │────────────────│
│ kingdom_id  │──────< │ kingdom          (FK → Kingdom)      │ >──────│ kingdom        │
│ kingdom_name│        │ population_id    (FK → Population)   │        │ symbol         │
└─────────────┘        │ generation_id    (FK → Generation)   │        │ symbol_name    │
                       │ individual_id                        │        │ alias          │
┌─────────────┐        │ chromosome_id                        │        │                │
│ POPULATION  │        │ gene_seq_id                          │        │ in_*  (arities)│
│─────────────│        │ symbol           (FK → Symbol Table) │        │ out_* (arities)│
│population_id│──────< │ ishead                               │        └────────────────┘
│ kingdom_id  │        │ botji_addr                           │
└─────────────┘        │ raw_text                             │        ┌────────────────┐
                       │                                      │        │ TERMINAL TABLE │
┌─────────────┐        │                                      │ >──────│────────────────│
│ GENERATION  │        │                                      │        │ kingdom        │
│─────────────│        │                                      │        │ symbol  (<0)   │
│generation_id│──────< └──────────────────────────────────────┘        │ symbol_name    │
│population_id│                                                         │ alias          │
└─────────────┘                                                         │ value          │
                                                                        │ out_* (arities)│
                                                                        └────────────────┘
```

**GeneFrame is the transaction table** — the central fact in a star schema.
It connects the evolutionary context (Kingdom → Population → Generation) with
the genetic vocabulary (Symbol Table, Terminal Table) through individual gene
positions.

---

## 2. The Many-Hot Arity Encoding

This is the core innovation over classical GEP.

In classical GEP, a symbol has a single integer `arity`. That integer captures
how many arguments the function takes — but nothing about *what types* those
arguments are.

In nucleotable, arity is **many-hot encoded** across typed columns:

```
Classical GEP:            nucleotable:
  symbol | arity    →     symbol | in_ORG | in_GPE | in_MONEY | in_NP | out_RELATION | ...
  ───────┼──────          ───────┼────────┼────────┼──────────┼───────┼──────────────┼────
  acquire|   3            acquire|   2    |   0    |    1     |   1   |      1       | ...
  visit  |   3            visit  |   0    |   1    |    0     |   0   |      0       | ...  
  strike |   5            strike |   1    |   1    |    0     |   0   |      0       | ...
```

`acquire` has `in_ORG=2` (takes two ORG arguments), `in_MONEY=1` (takes one
MONEY argument), `in_NP=1` (takes one noun phrase). Total arity = 4. But the
*type signature* tells you far more than the integer 4 does:

- Type-safe crossover is possible: a gene segment that outputs `out_ORG=1`
  can only be spliced into a slot where `in_ORG >= 1`. Evolution cannot produce
  programs that pass a GPE into a MONEY slot.
- MaxArity is computable: `SELECT MAX(in_ORG + in_GPE + in_MONEY + in_NP + …)`
- TailLength is derivable per kingdom: `HeadLength × (MaxArity - 1) + 1`
- Kingdoms extend naturally: adding a new type is adding a column, not
  changing the schema structure.

### 2.1 Many-hot vs one-hot

One-hot encoding: exactly one column is 1, all others 0.  
Many-hot encoding: **multiple columns can be non-zero simultaneously.**

The arity columns are many-hot because a single function can consume multiple
different types in the same call:

```
acquire:  in_ORG=2, in_MONEY=1, in_NP=1  ← four non-zero columns
```

The arity value (0, 1, 2, 3…) is the *count* for that type slot — not a flag.
It's a typed arity vector, not a label vector.

---

## 3. Entity Definitions

### 3.1 Kingdom

The genetic domain — defines which symbol set is in play.

```sql
CREATE TABLE kingdoms (
    kingdom_id    INTEGER PRIMARY KEY,
    kingdom_name  VARCHAR UNIQUE,   -- 'NLP-English', 'SQL', 'BotjiKingdom', ...
    description   VARCHAR,
    version       VARCHAR
);
```

Examples: `Symbolic Regression`, `SQL`, `REGEX`, `NLP-English`, `BotjiKingdom`, `Directive`.

A kingdom defines:
- Which symbols exist (rows in `symbols` with this `kingdom_id`)
- Which terminals exist (rows in `terminals`)
- The type columns relevant to this domain

---

### 3.2 Population

A named group of individuals sharing a common evolutionary context and kingdom.

```sql
CREATE TABLE populations (
    population_id  VARCHAR PRIMARY KEY,  -- 'bbc2026', 'reuters2026', 'sr-exp-001'
    kingdom_id     INTEGER REFERENCES kingdoms(kingdom_id),
    description    VARCHAR,
    created_at     TIMESTAMP
);
```

For news: `population_id = 'bbc2026'` (BBC, 2026 editorial era).  
For symbolic regression: `population_id = 'sr-run-20260311'`.

---

### 3.3 Generation

A temporal or evolutionary step within a population.

```sql
CREATE TABLE generations (
    population_id  VARCHAR REFERENCES populations(population_id),
    generation_id  INTEGER,             -- YYYYMMDD for news; 0,1,2... for GEP
    generation_type VARCHAR,            -- 'observed' | 'simulated'
    PRIMARY KEY (population_id, generation_id)
);
```

For news: `generation_id = 20260128` (date).  
For GEP: `generation_id = 42` (42nd evolutionary step).  
`generation_type` distinguishes real observations from simulated projections.

---

### 3.4 Symbol Table

The vocabulary of functions for a kingdom. Non-terminals: nodes with children.

```sql
CREATE TABLE symbols (
    kingdom_id     INTEGER REFERENCES kingdoms(kingdom_id),
    symbol         INTEGER,             -- positive IDs; unique within kingdom
    symbol_name    VARCHAR,
    alias          VARCHAR,             -- target expression language name

    -- Input arities (many-hot encoded — 0 means type not accepted)
    -- Base types (Symbolic Regression / SQL / REGEX kingdoms)
    in_S           INTEGER DEFAULT 0,   -- String
    in_I           INTEGER DEFAULT 0,   -- Integer
    in_F           INTEGER DEFAULT 0,   -- Float
    in_B           INTEGER DEFAULT 0,   -- Boolean
    in_A           INTEGER DEFAULT 0,   -- Array
    in_L           INTEGER DEFAULT 0,   -- List

    -- NLP-English types (NLP kingdom)
    in_PERSON      INTEGER DEFAULT 0,
    in_ORG         INTEGER DEFAULT 0,
    in_GPE         INTEGER DEFAULT 0,
    in_LOC         INTEGER DEFAULT 0,
    in_NORP        INTEGER DEFAULT 0,
    in_FAC         INTEGER DEFAULT 0,
    in_EVENT       INTEGER DEFAULT 0,
    in_PRODUCT     INTEGER DEFAULT 0,
    in_DATE        INTEGER DEFAULT 0,
    in_TIME        INTEGER DEFAULT 0,
    in_MONEY       INTEGER DEFAULT 0,
    in_QUANTITY    INTEGER DEFAULT 0,
    in_CARDINAL    INTEGER DEFAULT 0,
    in_PERCENT     INTEGER DEFAULT 0,
    in_NP          INTEGER DEFAULT 0,
    in_AP          INTEGER DEFAULT 0,
    in_CLAUSE      INTEGER DEFAULT 0,
    in_VERB        INTEGER DEFAULT 0,

    -- BotjiKingdom types
    in_ADDR        INTEGER DEFAULT 0,   -- botji address string

    -- Output arities (many-hot encoded)
    -- Base types
    out_S          INTEGER DEFAULT 0,
    out_I          INTEGER DEFAULT 0,
    out_F          INTEGER DEFAULT 0,
    out_B          INTEGER DEFAULT 0,
    out_A          INTEGER DEFAULT 0,
    out_L          INTEGER DEFAULT 0,

    -- NLP-English Phylo output types
    out_ENTITY     INTEGER DEFAULT 0,   -- 📦
    out_RELATION   INTEGER DEFAULT 0,   -- 🔗
    out_METRIC     INTEGER DEFAULT 0,   -- 📊
    out_EVENT      INTEGER DEFAULT 0,   -- ⚡
    out_PROCEDURE  INTEGER DEFAULT 0,   -- ⚙️
    out_NARRATIVE  INTEGER DEFAULT 0,   -- 📜

    -- BotjiKingdom output types
    out_ADDR       INTEGER DEFAULT 0,

    PRIMARY KEY (kingdom_id, symbol)
);
```

---

### 3.5 Terminal Table

The vocabulary of leaf nodes — typed constants and variables.

```sql
CREATE TABLE terminals (
    kingdom_id    INTEGER REFERENCES kingdoms(kingdom_id),
    symbol        INTEGER,    -- NEGATIVE IDs distinguish terminals from symbols
    symbol_name   VARCHAR,
    alias         VARCHAR,
    value         VARCHAR,    -- literal value (for constants)

    -- Output arities only (terminals are leaves — they produce, never consume)
    -- Same columns as symbols.out_* — but only output columns are populated
    out_S         INTEGER DEFAULT 0,
    out_I         INTEGER DEFAULT 0,
    out_F         INTEGER DEFAULT 0,
    -- ... (all out_* columns)
    out_PERSON    INTEGER DEFAULT 0,
    out_ORG       INTEGER DEFAULT 0,
    out_GPE       INTEGER DEFAULT 0,
    -- ... (all NLP out_* columns)
    out_ADDR      INTEGER DEFAULT 0,

    PRIMARY KEY (kingdom_id, symbol)
);
```

**Convention:** Terminal `symbol` values are negative integers. This allows a
single geneframe row to reference either a function (`symbol > 0`) or a terminal
(`symbol < 0`) with no ambiguity, and a single JOIN across both tables via UNION.

---

### 3.6 GeneFrame (transaction table)

The central fact table. One row per gene position per chromosome.

```sql
CREATE TABLE geneframe (

    -- Exogenetic context (project/experiment metadata)
    user_name      VARCHAR,
    project        VARCHAR,
    experiment_id  INTEGER,

    -- Evolutionary context (the population hierarchy)
    population_id  VARCHAR REFERENCES populations(population_id),
    generation_id  INTEGER,
    island_id      INTEGER DEFAULT 0,   -- for distributed island models
    individual_id  VARCHAR,             -- topic/individual within the population
    birth_gen      INTEGER,             -- generation this individual was born
    age            INTEGER,             -- generations this individual has survived

    -- Genetic structure
    chromosome_id  VARCHAR,             -- article URL, UUID, or hash
    kingdom_id     INTEGER REFERENCES kingdoms(kingdom_id),
    ishead         BOOLEAN,             -- TRUE = head region; FALSE = tail (non-coding buffer)
    gene_seq_id    INTEGER,             -- position within chromosome (BFS order)
    symbol         INTEGER,             -- FK → symbols (>0) or terminals (<0)

    -- Botji address (derived, for BotjiKingdom operations)
    botji_addr     VARCHAR,             -- 'population/generation/individual/chromosome'

    -- Source (for news/text corpora)
    raw_text       VARCHAR,             -- original sentence text
    source_url     VARCHAR              -- article URL
);
```

---

## 4. MaxArity and TailLength

These are computed from the symbol table at query time, per kingdom:

```sql
-- MaxArity for kingdom_id = 3 (NLP-English)
SELECT MAX(
    in_PERSON + in_ORG + in_GPE + in_LOC + in_NORP + in_FAC +
    in_EVENT + in_PRODUCT + in_DATE + in_TIME + in_MONEY +
    in_QUANTITY + in_CARDINAL + in_PERCENT + in_NP + in_AP +
    in_CLAUSE + in_VERB + in_ADDR
) AS max_arity
FROM symbols
WHERE kingdom_id = 3;

-- TailLength
-- TailLength = HeadLength × (MaxArity - 1) + 1
```

This query works for *any* kingdom because all type columns default to 0.
Adding a new kingdom with new type columns doesn't require changing the query —
only the column list in the SUM.

---

## 5. Type-Safe Crossover Query

The many-hot encoding enables a SQL-expressible type compatibility check for
crossover operations:

```sql
-- Find compatible splice points between two chromosomes
-- (positions where output type of gene A matches input type of gene B's parent)
SELECT
    a.gene_seq_id  AS splice_point_a,
    b.gene_seq_id  AS splice_point_b
FROM geneframe a
JOIN geneframe b
    ON  a.chromosome_id  = :chrom_a
    AND b.chromosome_id  = :chrom_b
JOIN symbols sa ON sa.symbol = a.symbol AND sa.kingdom_id = a.kingdom_id
JOIN symbols sb ON sb.symbol = b.symbol AND sb.kingdom_id = b.kingdom_id
WHERE
    -- Output of a's gene is compatible with input slot of b's parent
    (sa.out_ORG  > 0 AND sb.in_ORG  > 0) OR
    (sa.out_GPE  > 0 AND sb.in_GPE  > 0) OR
    (sa.out_NP   > 0 AND sb.in_NP   > 0) OR
    (sa.out_ADDR > 0 AND sb.in_ADDR > 0) OR
    -- ... (all type pairs)
    (sa.out_F    > 0 AND sb.in_F    > 0);
```

---

## 6. Relationship to the BotJi Protocol

The data model is the **wire format** of the BotJi protocol.

A BotJi message is a set of geneframe rows — a chromosome, exchangeable as:
- JSON Lines (streaming)
- Parquet (batch / peer sync)
- Arrow IPC (in-memory transfer between WASM modules)

The botji address is the message key. The symbol column is the content,
resolved against the shared symbol table (the BotJi vocabulary).

Two agents that share a kingdom definition (the same `symbols` and `terminals`
tables) can exchange chromosomes and decode them identically. The kingdom is
the shared schema; the geneframe rows are the messages.

---

## 7. Summary of the nucleotable Innovation

| Classical GEP | nucleotable |
|---------------|-------------|
| Gene = position in a linear string | Gene = **row** in a relational table |
| Genome = object in memory | Genome = **set of rows** in a dataframe |
| Population = list of objects | Population = **partition of a table** |
| Arity = single integer | Arity = **many-hot vector of typed counts** |
| Symbol table = dict/array | Symbol table = **relational dimension table** |
| Evolution = in-process loop | Evolution = **SQL queries** |
| Exchange = serialise objects | Exchange = **Parquet / Arrow** |
| Single-type functions | **Multi-typed functions** with full type signatures |

The Karva language (Ferreira) defines the rules.  
nucleotable gives it a data model that scales.
