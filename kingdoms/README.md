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

Since 2026-10-20 every kingdom runs on one design: true multi-typed GEP
(`nucleotable/docs/MultiTypedGEP.md` rev 4). It gives a head and a tail for
every output type, ordered argument types, and no repair or fallback anywhere,
so every gene expresses a legal program.

| folder | status (2026-10-23) | what it evolves |
|---|---|---|
| `symbolic-regression/` | **live** | the general float-typed op set; byte-identical on the typed core (481 of 1,000 seed-runs on 100 laws, old and new) |
| `tsr/` | **built** | **TSR — Transcendental Symbolic Regression.** The same op set with the distance between functions TYPED, so illegal depth is unrepresentable rather than penalised. Tables in `gen_tables.py`, arms in `single/`, `depth/`, `order/`, `order_depth/`; the pre-registered experiments are not yet run |
| regex (tables in code) | **live** | regular expressions, four types (Pattern, CharClass, Char, Integer). UK postcodes: a perfect training rule in 32 of 32 typed runs, held-out recall 100 %, no foreign codes accepted (phylu `docs/reports/2026-10-20_regex_typed_gep.md`) |
| `wgsl_true/` | **live** | **WGSL GPU kernels, the true multi-typed kingdom.** Every legality rule is carried by a type (ownership, uniform control, bounded loops, ranges). Instances: the Gram test, the cuneus shaders, the UltraDim ingest module. See `wgsl_true/README.md` |
| `wgsl/`, `safewgsl/` | legacy | the earlier WGSL kingdoms (dual/class tables, a reader with refusals); kept while their tests are ported, then retired |
| `sql/` | proposed | SQL statements — the original nucleotable motivation: reverse-engineering running systems |
| `brainfuck/` | proposed | a minimal language, chosen because it is small enough to be a clean test of the harness itself |

## What each folder holds

```
<kingdom>/
  README.md        what this kingdom is for, and its typing rules
  gen_tables.py    the generator: the design written as code
  types.tsv        the types
  functions.tsv    the symbol table: output type, ordered argument types, spelling
  symbols.md       (older kingdoms) the human-readable symbol table
```

The table is the design; the Rust is its loader. A kingdom check (rev 4 §1)
runs on every generated table. It checks:
- one output type per symbol;
- ordered, contiguous signatures;
- a terminal for every demanded type.

## The harness claim, stated so it can be tested

The claim is that **one engine serves all of these** — the same selection, the
same pump, the same islands, the same HFF — and only the symbol table changes.
Where it stands (2026-10-23):
- SR, regex and WGSL all run on the one typed core. Selection, crossover,
  mutation, the pump and HFF are shared, and only the tables differ.
- Legality holds in all three: every random gene expresses a legal program.
- Learning differs. SR and regex learn their problems. WGSL legal kernels
  evolve, and planted slowdowns are found and removed, but no real kernel has
  yet been made faster than its hand-written original.
- WGSL needed engine work beyond its table:
  - SR's in-loop optimisers, ported;
  - two-parent marriage;
  - a two-level HFF for quality against speed;
  - GPU work guards.
- All of these are general and now serve every kingdom. Whether that counts
  against the harness claim is an open question.

Shared definitions belong to the geneframe, not to a kingdom (rev 4 §6):
- Population-wide shared subtrees are **ADFs**. Each is an immutable
  expressed tree with a stable id, a root type and a level, referenced by a
  typed terminal `A_d`.
- A chromosome's genes are joined by a **homeotic gene**: Ferreira's linker,
  fixed or evolved. In WGSL it links a module's entry points.
- Fold turns shared subtrees into ADFs, and unfold inlines them. Values are
  identical either way, a tested contract.

SR expressions, WGSL kernels and SQL or Spark plans instantiate that one
mechanism.

## A note on scope

LLMs have changed what SQL reverse-engineering is worth. The design is still
valid and the kingdom is still a good test of the harness — it needs relations
and predicates, which SR does not, so it exercises the typing system in a way
float-only work cannot.
