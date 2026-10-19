# Multi-typed GEP: one head, one tail per type

Design, 2026-10-19. Finishes `EvolutionSQL.md` §8 ("the typed tail", left open there) and
`DataModel.md` §2 (many-hot typed arities). Source: Andrew's design. Not built.

## Context

**Objective.** Multi-typed gene expression programming in which it is impossible to express an
invalid tree: every expressed node has the type its parent's slot demands, for every gene, every
mutation, every crossover. This is GEP's promise (Ferreira: "their expression always results in
syntactically correct programs") carried to many types.

**Approach.** One head, plus one tail per type. A slot demanding type T is filled from T's own
reading frame: the next unused head row that outputs T, else the next row of T's tail. Tails hold
only terminals of their own type, so no slot can ever be filled by a symbol of the wrong type.

**Unique feature.** No repair, no rejection, no substitution and no stand-in (fallback) symbol
anywhere. Validity comes from the layout, as in classical GEP and as GEP-RNC's Dc domain fills
its `?` placeholders. A literature search (2026-10-19: arXiv, Semantic Scholar, Crossref,
OpenAlex, GECCO/EuroGP) found no published GEP with a tail per type; published multi-type
methods restrict operators (Montana's STGP), reject and retry (Zhou et al. 2003), filter
(DHC-GEP 2024), or use typed graph wiring (typed CGP).

**Status.** Design only. The built system (fuller `safe_karva.rs` projection, phylu
`decode.wgsl`) does the opposite: one shared tail, and a symbol of the wrong type is replaced by
a fallback leaf. That is wrong and is removed by this plan (§9).

**Why.** Every stand-in is a hidden defect: it changes the program the gene encodes, makes
crossover and mutation produce children with holes, and broke the WGSL runs of 2026-10-19
(14 % of mutants gained stand-ins; genes expressed with no statements).

## 1. The symbol table (unchanged from DataModel.md)

Each function row carries typed input counts `in_T` and **exactly one** output type `out`.
Each terminal row carries one output type and no inputs.

- A polymorphic function (one name, several output types, e.g. `match` giving a string or an
  integer) is one row per output type. A row never has two outputs: one row, one type, so the
  reading frame of every type is well defined.
- Kingdom check (run when a kingdom is loaded, a failure is a kingdom bug): every type that is
  the root type or appears in any `in_T` has at least one terminal.

Notation: types T; head length h; `a_T` = the largest `in_T` of any function; `n_max` = the
largest total arity; R = the root type of the gene.

## 2. The chromosome

| domain | length | may hold |
|---|---|---|
| head | h | any function or terminal of the kingdom, any type |
| tail of T, one per type | t_T | terminals of type T only |
| Dc of T (only for types with random constants) | t_T | indices into T's constant array |

Tail length per type, from a count of demands, not guessed:

- Functions are drawn only from the head, so at most h functions are expressed.
- Each expressed function opens at most `a_T` slots of type T; the root opens one slot of R.
- So at most `h·a_T + [T = R]` slots of type T are ever demanded, and also at most
  `h(n_max − 1) + 1` leaves in all (Ferreira's bound).

```
t_T = min( h·a_T + [T = R],  h(n_max − 1) + 1 )
```

A type never demanded (`a_T = 0` and not the root) has no tail. Single type: `a_T = n_max`, and
`t = h(n_max − 1) + 1` exactly, so this is classical GEP when there is one type.

## 3. Expression (deterministic, level order)

Expression reads the tree in level order (Karva: top to bottom, left to right), as Ferreira does.

1. Keep a cursor per type: `k_T` = how many slots of type T have been filled so far.
2. The root is a slot of type R.
3. For each slot of type T, in level order:
   - let `H_T` = the head rows whose output is T, in head order;
   - if `k_T < |H_T|`: the slot takes head row `H_T[k_T]` (a function or a terminal of T);
   - else it takes tail-T row `k_T − |H_T|`;
   - `k_T += 1`; if the symbol is a function, append its typed input slots in argument order.
4. Stop when no slot is open.

The tail is never exhausted (§2), and every node has its slot's type by construction. Head rows
of type T that are never reached are non-coding, exactly as an unexpressed head suffix is in
classical GEP.

As SQL, the k-th demand of type T is a rank within the type:

```sql
-- the reading frame of each type: its head rows first, then its tail, in gene order
SELECT type, gene_seq_id, symbol,
       ROW_NUMBER() OVER (PARTITION BY type ORDER BY is_tail, gene_seq_id) - 1 AS k
FROM geneframe g
JOIN symbol_types s USING (kingdom_id, symbol)   -- s.type = the row's one output type
WHERE g.chromosome_id = :chrom;
-- the level-order walk joins slot (type, k) to this frame: no slot can join a row of another type
```

## 4. Constants (GEP-RNC, per type)

A type with random constants has a placeholder terminal `?_T` in its alphabet and a Dc domain of
length t_T. The j-th expressed `?_T` (level order) takes the value `C_T[Dc_T[j]]`. This is
Ferreira's Dc unchanged, one per type. A literal is never a symbol of the table.

## 5. Variation, valid by construction

Every operator keeps each domain's alphabet: the head holds any kingdom symbol, the tail of T
holds terminals of T, Dc_T holds indices of C_T. Nothing more is needed, because expression
chooses by type.

| operator | rule |
|---|---|
| mutation | head position: any symbol of the kingdom (unconstrained, as Ferreira); tail-T position: any terminal of T; Dc_T: any index of C_T |
| inversion | inside one domain |
| IS / RIS transposition | a sequence from any domain, inserted only into the head (tail and Dc lengths unchanged) |
| one-point, two-point, gene recombination | homologous: the same positions of two aligned chromosomes of one kingdom and layout |
| constant mutation | a new value in C_T |

A change in the head can change which head rows are used (a function's type decides which frame
it is in). It never changes validity.

## 6. Multigenic chromosomes and homeotic genes

Each gene has its own head, its own tails and its own root type R_g. A homeotic gene's genic
terminals are terminals of their gene's root type, so a reference is type-checked like any other
terminal. A homeotic row's href names its stable row id (`PLAN_population_dag.md`).

## 7. The geneframe rows

`geneframe` gains two columns: `domain` (`head`, `tail`, `dc`) and `type` (the tail's or Dc's
type; NULL for head rows, whose type is their symbol's output). `ishead` stays for
compatibility (`domain = 'head'`). Initial population, mutation and recombination stay the SQL
of `EvolutionSQL.md`, with the tail draw being `random_terminal(:kingdom, :type)`.

## 8. The WGSL kingdom under this design

- Its duals are the types. Statement nesting levels (L0–L4, C0–C4, O) stay as types.
- Bindings, registers and array names of a kernel's interface are terminals of their duals.
- Literal duals use §4 (Dc per literal dual), not table rows.
- Language defaults (no epilogue, the default grid) are named terminals with their own text
  (`epilogue.none`, `grid.default`), never a generic fallback.
- Size: tails exist only for duals some function demands; `t_T = h·a_T + [T = R]` keeps rarely
  demanded duals short.

## 9. What is removed

Every substitution, in both crates:

- fuller `src/wgsl/safe_karva.rs`: `project_scored` (:988, :994, :998, :1008, :1031),
  `conform_at` (:748, :755, :851), the policy at :20;
- fuller `src/geneframe.rs` `fallback_leaf` / `fallback_table` (:179, :195), `loader.rs`
  fallback tables (:230, :242, :251);
- phylu `src/evolve/decode.wgsl` (:283–296, bindings 20 and 21), and its host callers;
- the reader's stand-ins for unread values (safe_reader :2177, :4197): a construct that does not
  read is a kingdom gap, reported, never a fallback.

The SR and regex kingdoms move to the same layout. SR is single typed, so for SR this design is
classical GEP and its results must not change.

## 10. Tests that certify it

1. **Closure.** For each kingdom (SR, regex, WGSL v24 interfaces): 100,000 random genes, plus
   100,000 children of each operator; every expressed node's type equals its slot's type,
   checked node by node; zero exceptions. Expression asserts and never substitutes.
2. **Bound.** The largest tail index used never reaches t_T; the measured maximum is reported
   beside the bound.
3. **Classical equality.** SR with one type: the expressed trees are identical to classical
   Karva decoding for the same genes.
4. **Determinism.** The same gene gives the same tree, on the host and on the GPU, bit for bit.
5. **Round trip.** Seeds read from WGSL text encode into this layout and express back to the
   same tree.

## 11. Build order

1. The layout, tail lengths and expression in fuller (host), with tests 1–4.
2. The GPU expression in phylu: per-type cursors, no fallback bindings; test 4.
3. Operators in phylu (§5), test 1 over the children.
4. The SR and regex kingdoms moved; their goldens re-checked.
5. The WGSL kingdom moved, the saved genomes re-encoded, test 5; then evolution runs.
