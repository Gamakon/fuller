# Kingdoms — a symbol table is a phylogenetic space

## The proposal

Andrew Morgan's design for nucleotable: **gene expression programming as a
general-purpose harness for building evolutionary computing languages.** A
kingdom is a symbol table with a typing system; the engine is the same
underneath. Change the table and you change what can evolve, without changing
the search.

> "The design of gene expression programming around a symbol table with types
> was my proposal to try to make gene expression programming a general purpose
> harness for designing evolutionary computing languages."

Each subfolder here is one kingdom: its symbol table, its typing rules, and
the measurements taken in it.

## Why types, and why they are not decoration

A K-expression's head may hold any symbol; the tail holds terminals; the arity
walk guarantees a valid tree. That guarantee is about SHAPE, not MEANING — so
an untyped table lets the search build `tanh(exp(cos(log(x))))` as readily as
`m*v**2/2`, and spends its head space on both.

Typing makes an illegal gene **unrepresentable** rather than penalised. The
first worked case is in `symbolic-regression/`: no SRBench true law nests one
transcendental directly inside another, measured across all 133, so a type
that cannot be nested removes those shapes from the space entirely rather than
detecting and folding them afterwards.

**The more general-purpose the target language, the more types it needs.** SR
needs two or three. SQL needs relations, columns and predicates. A full AST
language needs whatever its own type system has — which is why almost every
AST language ships one.

## The kingdoms

| folder | status | what it evolves |
|---|---|---|
| `symbolic-regression/` | **live** | the general float-typed op set the engine runs today |
| `tsr/` | **specified** | **TSR — Transcendental Symbolic Regression.** The same op set with the distance between functions TYPED, so illegal depth is unrepresentable rather than penalised |
| `sql/` | proposed | SQL statements — the original nucleotable motivation: reverse-engineering running systems |
| `regex/` | proposed | regular expressions; the typing system was sketched in the nucleotable work |
| `brainfuck/` | proposed | a minimal language, chosen because it is small enough to be a clean test of the harness itself |

## What each folder holds

```
<kingdom>/
  README.md      what this kingdom is for, and its typing rules
  symbols.md     the symbol table: semantic_id, alias, typed many-hot arity
  measurements/  what was measured in this kingdom, with the run cards
```

`symbols.md` is the human-readable statement of what `geneframe::master_table()`
builds for that kingdom, and the two must agree. The table is the design; the
Rust is its loader.

## The harness claim, stated so it can be tested

The claim is that **one engine serves all of these** — the same selection, the
same pump, the same islands, the same HFF — and only the symbol table changes.
It is not proven. SR is the only kingdom with a loaded table and measurements.
Whether the harness generalises is the thing the other kingdoms are for, and
the first one that needs a change to the ENGINE rather than to a table is the
one that falsifies it.

## A note on scope

LLMs have changed what SQL reverse-engineering is worth. The design is still
valid and the kingdom is still a good test of the harness — it needs relations
and predicates, which SR does not, so it exercises the typing system in a way
float-only work cannot.
