# nucleotable · GeneFrame
### A dataframe for artificial genetic programming — and a formal model of how information evolves

---

## The Founding Insight

Some languages are read horizontally. Some are read vertically.

Classical GEP represents chromosomes as **horizontal lists** — a sequence of
symbols read left to right, like a sentence: `[ +, *, x, y, 3, x, y, 2, 1 ]`

nucleotable asks: *what if you read each gene vertically?*

**Rotate the chromosome 90 degrees. Stack many chromosomes. You have a database.**

The sequence becomes a column value (`gene_seq_id`). Position is data, not
structure. And suddenly the entire ecosystem of SQL, DuckDB, Parquet, and WASM —
built for exactly this shape of data — becomes available to GEP.

See [`docs/design/Philosophy.md`](docs/design/Philosophy.md) for the full argument.

---

## What is this?

**nucleotable** is a DuckDB-backed dataframe for [Gene Expression Programming (GEP)](https://en.wikipedia.org/wiki/Gene_expression_programming).
It extends classical GEP with:

- **Multi-typed symbols** — functions carry full typed input/output signatures (`in_S, in_I, in_F…`), not just a single arity integer. Type-safe crossover: genes only splice where types are compatible.
- **Multi-kingdom populations** — Symbolic Regression, SQL, REGEX, NLP-English, and BotjiKingdom live in the same table, differentiated by `kingdom`.
- **SQL-native evolution** — initial population generation, mutation, crossover, and fitness evaluation are all SQL queries over DuckDB. No objects in memory; populations are relational rows.
- **Botji addressing** — every gene has a four-part hierarchical address (`population/generation/individual/chromosome`) that uniquely identifies it in the corpus.

---

## The OODA Intelligence Loop

The system implements a closed intelligence loop over live news data:

```
OBSERVE  →  ORIENT  →  DECIDE  →  ACT
```

| Phase | What happens |
|-------|-------------|
| **Observe** | Fetch RSS feeds → encode sentences in Karva via `lang2karva` WASM → append to staging geneframe |
| **Orient** | Deduplicate (Simhash), cluster (Streaming KMeans), track story mutations and causality — produce refined geneframe |
| **Decide** | Fit evolutionary operators from observed generation-to-generation transitions → simulate population forward 30 days |
| **Act** | Emit a directive as a Karva expression — stored back in the geneframe, closing the loop |

See [`docs/design/OODAArchitecture.md`](docs/design/OODAArchitecture.md) for the full architecture.

---

## Botji Address

A **botji address** is a four-part coordinate that identifies any unit of content:

```
<book> / <chapter> / <paragraph> / <verse>
```

For news:

```
bbc2026 / 20260128 / world / bbc-68123456
   │           │        │          │
source+era   date    topic     article
```

A botji address **is** a geneframe address:

| Botji | GeneFrame key | Example |
|-------|--------------|---------|
| Book | `population_id` | `bbc2026` |
| Chapter | `generation_id` | `20260128` |
| Paragraph | `individual_id` | `world` |
| Verse / sentence | `chromosome_id` + `gene_seq_id` | `bbc-68123456` + `3` |

---

## BBC Test Data Example

The following shows how a single BBC article sentence is encoded into the geneframe.

**Source article:**
```
BBC World News · 2026-01-28
"Iran launched a drone attack on a US military base in Jordan, killing three soldiers."
URL: https://www.bbc.co.uk/news/world-middle-east-68123456
```

**Botji address:** `bbc2026/20260128/world/bbc-68123456`

**NLP-English encoding** (`lang2karva` WASM output):

The dependency parse root is `launch` → matched to symbol `strike` (ID=10, `out_EVENT=1`).
BFS traversal produces:

```
HEAD:   strike          [func, symbol=10, arity=5]
TAIL:   Iran            [GPE,  symbol=-301]
        Jordan          [GPE,  symbol=-302]
        US_military_base [FAC, symbol=-303]
        American        [NORP, symbol=-304]
        three_soldiers  [NP,   symbol=-305]
```

HeadLength=1, MaxArity=5, TailLength = 1×(5-1)+1 = 5 → 6 gene rows total.

**GeneFrame rows:**

```
population_id | generation_id | individual_id | chromosome_id  | gene_seq_id | kingdom     | ishead | symbol | symbol_name     | raw_text                                          | botji_addr
bbc2026       | 20260128      | world         | bbc-68123456   | 0           | NLP-English | TRUE   | 10     | strike          | Iran launched a drone attack on a US military...  | bbc2026/20260128/world/bbc-68123456
bbc2026       | 20260128      | world         | bbc-68123456   | 1           | NLP-English | TRUE   | -301   | Iran            | Iran launched a drone attack on a US military...  | bbc2026/20260128/world/bbc-68123456
bbc2026       | 20260128      | world         | bbc-68123456   | 2           | NLP-English | TRUE   | -302   | Jordan          | Iran launched a drone attack on a US military...  | bbc2026/20260128/world/bbc-68123456
bbc2026       | 20260128      | world         | bbc-68123456   | 3           | NLP-English | FALSE  | -303   | US_military_base| Iran launched a drone attack on a US military...  | bbc2026/20260128/world/bbc-68123456
bbc2026       | 20260128      | world         | bbc-68123456   | 4           | NLP-English | FALSE  | -304   | American        | Iran launched a drone attack on a US military...  | bbc2026/20260128/world/bbc-68123456
bbc2026       | 20260128      | world         | bbc-68123456   | 5           | NLP-English | FALSE  | -305   | three_soldiers  | Iran launched a drone attack on a US military...  | bbc2026/20260128/world/bbc-68123456
```

**Phylo output (decoded):**
```
⚡  Iran  STRUCK  Jordan  :at US_military_base  :victims three_soldiers
```

A full article with 12 sentences produces ~72 geneframe rows (average 6 genes per sentence). A BBC world feed with 20 articles produces ~1,440 rows per generation (day).

---

## Design Documents

| Document | Description |
|----------|-------------|
| [`docs/design/BotjiAddressDesign.md`](docs/design/BotjiAddressDesign.md) | Botji address scheme, geneframe key mapping, evolutionary simulation goal, Locke/Descartes framing |
| [`docs/design/BotjiKingdomDesign.md`](docs/design/BotjiKingdomDesign.md) | BotjiKingdom symbol set — address navigation, relational predicates, generation arithmetic |
| [`docs/design/NLPKarvaDesign.md`](docs/design/NLPKarvaDesign.md) | NLP-English kingdom — dep tree BFS → Karva encoding |
| [`docs/design/NLPKarvaGeneSymbolTable.md`](docs/design/NLPKarvaGeneSymbolTable.md) | Full NLP-English symbol table with typed arity signatures |
| [`docs/design/GeneFrameDesign.md`](docs/design/GeneFrameDesign.md) | Initial population generation via SQL |
| [`docs/design/EvolutionSQL.md`](docs/design/EvolutionSQL.md) | Tournament selection, crossover, mutation — all as SQL CTEs |
| [`docs/design/Philosophy.md`](docs/design/Philosophy.md) | The manifesto — horizontal vs vertical, the founding insight, the news globe visualisation |
| [`docs/design/OODAArchitecture.md`](docs/design/OODAArchitecture.md) | OODA loop architecture — staging/refined graphs, four-phase intelligence loop |
| [`docs/design/NewsIngestionPipeline.md`](docs/design/NewsIngestionPipeline.md) | BBC RSS ingestion pipeline — fetch, address, encode, write, subscribe |

---

## Open Change Requests (CRs)

### CR-001 · Schema: population/individual/chromosome must be VARCHAR

The current `schema.py` declares `population_id`, `individual_id`, `chromosome_id` as `INTEGER`.
For news use, these are strings (`"bbc2026"`, `"world"`, `"https://bbc.co.uk/..."`).

**Required change:**
```sql
-- schema.py geneframe table
population_id  VARCHAR    -- was INTEGER
individual_id  VARCHAR    -- was INTEGER  
chromosome_id  VARCHAR    -- was INTEGER
kingdom        VARCHAR    -- was INTEGER — must match 'NLP-English', 'BotjiKingdom', etc.
```

### CR-002 · Schema: add botji_addr and raw_text columns

The geneframe needs two new columns for news encoding:

```sql
botji_addr  VARCHAR   -- full 4-part address: "bbc2026/20260128/world/bbc-68123456"
raw_text    VARCHAR   -- original sentence text (enables re-encoding without re-fetch)
```

`raw_text` in the staging geneframe means the symbol table can be updated and
historical rows re-encoded without re-fetching source articles.

### CR-003 · WASM: lang2karva must accept botji context and emit full geneframe rows

Currently `lang2karva` takes text and returns Karva tokens. For the ingestion
pipeline, it needs to:

1. **Accept botji context parameters:**
   ```
   lang2karva encode \
     --text "Iran launched a drone attack..." \
     --population bbc2026 \
     --generation 20260128 \
     --individual world \
     --chromosome bbc-68123456 \
     --gene-offset 0
   ```

2. **Emit geneframe rows** in a structured format (JSON Lines or CSV):
   ```jsonl
   {"population_id":"bbc2026","generation_id":20260128,"individual_id":"world","chromosome_id":"bbc-68123456","gene_seq_id":0,"kingdom":"NLP-English","ishead":true,"symbol":10,"symbol_name":"strike","raw_text":"Iran launched...","botji_addr":"bbc2026/20260128/world/bbc-68123456"}
   {"population_id":"bbc2026","generation_id":20260128,"individual_id":"world","chromosome_id":"bbc-68123456","gene_seq_id":1,"kingdom":"NLP-English","ishead":true,"symbol":-301,"symbol_name":"Iran","raw_text":"Iran launched...","botji_addr":"bbc2026/20260128/world/bbc-68123456"}
   ...
   ```

3. **Output can pipe directly into DuckDB:**
   ```bash
   lang2karva encode --text "..." --population bbc2026 ... \
     | duckdb geneframe.duckdb "COPY geneframe FROM '/dev/stdin' (FORMAT JSON)"
   ```

### CR-004 · BBC Fetcher: botji fetch bbc

A new CLI tool / Python module that:
1. Polls all BBC RSS feeds (see `NewsIngestionPipeline.md` §2.1)
2. Assigns botji addresses from RSS item metadata
3. Calls `lang2karva` per sentence
4. Writes to geneframe (idempotent — skip known `chromosome_id`)
5. In subscribe mode: polls on interval, appends new articles continuously

---

## Observational and Evolvable

This system extends work first published in:

> Morgan et al., *Mastering Spark for Data Science*, Chapter 10:  
> *"Story De-duplication and Mutation"*

That chapter is **observational** — it describes story mutation empirically
using Simhash, Streaming KMeans, and Kibana, validated against the Paris Attacks
of November 2015.

This system is **evolvable** — it encodes the same dynamics formally in GEP,
enabling simulation forward in time from fitted evolutionary operators.

John Locke observed the world and described what he saw.
René Descartes derived the world from first principles and explored what could be.
Both are right. Both are necessary.

**One describes. The other explores.**

---

## Kingdoms

| Kingdom | What it evolves | Symbol types |
|---------|----------------|-------------|
| Symbolic Regression | Mathematical expressions | Float in/out |
| SQL | SQL expressions over tabular data | S, I, F, B in/out |
| REGEX | Regular expression patterns | S, I in/out |
| NLP-English | Typed Phylo clauses from natural language | 17 NER/dep types in, 6 Phylo types out |
| BotjiKingdom | Structural navigation over hierarchical addresses | ADDR, S, I, B in/out |

---

## Quick Start

```bash
pip install -e .
```

```python
import duckdb
from nucleotable.db.connection import DuckDBConnection
from nucleotable.db.schema import create_schema
from nucleotable.db.kingdoms import load_kingdoms

db = DuckDBConnection('data/geneframe.duckdb')
conn = db.connect()
create_schema(conn)
load_kingdoms(conn)
```
