# Prompt: translate a WGSL kernel into a safeWGSL GEP chromosome

Status: version 4, DRAFT for review (2026-10-19). Version 4 adds what MiniMax M3's and ChatGPT's
version 2 replies showed: which loop row a `for` is, the epilogue's grouping from the front, literals in
the form demanded, and a reply that begins with its gene.
Version 3 (fuller 0c33257): Version 1 (fuller 46bccc7) and version 2 (fuller a834d20)
are stored byte for byte in tests/fixtures/gep_translate/ with the replies they drew. Version 3 asks for Karva,
not an s-expression: a reply always decodes to a valid gene (no brackets to miscount), its open reading
frame right-padded with random valid terminals. Part A is the fixed header (the same for every call on
one kingdom and problem, so it is the cached prefix); Part B is the per-call challenge. Text
between `{{` `}}` is filled by the prompt builder from the kingdom's own tables, never typed
by hand. The design notes at the end are for us, not sent to the model.

---

## PART A — the fixed header (cached)

### A1. What you are doing

You are a translator for a gene expression programming (GEP) system that evolves GPU compute
kernels. You will be given a program and asked to write it as a GEP gene in Karva notation
over a fixed, typed symbol table. You do not write WGSL. You do not improve the program. You
write the genotype whose expression is this program, using only the symbols in the table
below.

### A2. Gene expression programming, as this system uses it

- **Genotype and phenotype.** An individual's genotype is a chromosome: a fixed-length string
  of symbols. Its phenotype is the expression tree the string expresses: a symbol with arity
  *n* takes the next *n* subtrees as its children. Here the phenotype is a GPU kernel, and the
  system renders it to WGSL itself.
- **Validity by construction.** Every symbol is a row of the symbol table with a typed
  signature: the type of each input and the type of its output. A child may sit in an input
  only if the child's output type is exactly that input's type. Because every operator of the
  system respects these types, every genotype expresses a valid program. There is no
  "compile error" in this system: a tree that breaks a typing rule is a mistranslation.
- **Strongly typed, multityped.** There are many types (listed in A5). Each row's arity is
  fixed by the table (at most 4 inputs). Terminals have arity 0.
- **Shared subtrees.** You write the whole tree. Where the same subtree occurs many times,
  write it each time; the system finds repeats and folds them into homeotic tail genes itself.

### A2b. Karva: how a gene is written and read

A gene is a flat list of symbols, with no brackets. It is read **level by level, left to
right**:
1. The first symbol is the root.
2. The next symbols are the root's children, as many as its arity, left to right.
3. Then come the children of those children, level by level. Within a level the order is
   left to right: first all children of the level's first symbol, then all of the second's,
   and so on.

A symbol of arity 0 has no children. The reading stops when every symbol read has all its
children: that prefix is the **open reading frame (ORF)**. Nothing after it is expressed.

Example: the tree `a(b(d, e), c(f))`, with arities a 2, b 2, c 1, d 0, e 0, f 0, is the
Karva gene `a b c d e f`: level 0 is `a`; level 1 is `b c`; level 2 is `d e f` (the children
of `b`, then of `c`).

To write a gene from a tree: list the root; then, for each level in turn, list the children
of every symbol of the previous level, in that symbol's order and in the level's order. This
is a breadth-first walk of the tree.

**Right padding.** After the ORF, append a tail of terminals (arity-0 symbols): exactly
3 × F + 1 of them, where F is the number of symbols of arity 1 or more in your ORF. GEP's
tail rule guarantees that the gene decodes even if you misjudged an arity: a reading that
needs more children finds them in the tail. So the tail must be **varied, valid terminals
drawn from across the table**:
- literals of many forms, each with a value in its form's range;
- registers of each kind, builtins, the problem's uniform fields, counters;
- the statement terminals (`stmt.return_L0`, `stmt.barrier_L0`).

Choose them as a random-looking mixture: never `fallback`, never the same symbol twice in a
row, never a long run of one kind. A decoder that falls into the tail then expresses a
well-formed program, not padding.

**Typing while reading.** The decoder knows each position's demanded type from its parent's
signature. A symbol whose output type is not the demanded type is a mistranslation (the
decoder substitutes the type's fallback leaf and reports it), so write each symbol at a
position whose type it produces.

### A3. The kingdom: safeWGSL

The kingdom's symbols describe a whole compute kernel, statements and expressions alike.
Shapes in this section are written as trees with brackets, `(parent child …)`, only to show
their structure. You return Karva (A2b, A9).

- **The root** is `kernel.module_N` for a module of N entry points, each child a
  `kernel.entry_safe` (one entry point). An entry point is a grid (its workgroup size and the
  workgroup arrays it declares) and a body (a block of statements).
- **Statements** are rows too: assignments to registers (`reg.set_*`), private arrays
  (`parr.set_*`), workgroup arrays (`wg.put_*`), stores to output buffers (`store.*`), atomics,
  `if`, loops, barriers, return. Blocks (`block.seqN_*`, `block.one_*`) sequence 1–4
  statements; longer sequences nest.
- **An entry point has exactly three children:** `(kernel.entry_safe GRID BODY EPILOGUE)`.
  - GRID is the workgroup size and the workgroup arrays.
  - BODY is a `block.*_L0`.
  - EPILOGUE is a `block.*_O0`, or the terminal `fallback`.

  The epilogue rule:
  - **The last top-level statement holds the entry point's last store** to an output
    buffer, with nothing after that store. That store may sit inside `if`s and barrier-free
    loops. Then that statement is the epilogue, and the body is everything before it.
  - **An epilogue block** at depth d is `block.one_Od`, `block.seq2_Od`, `block.seq3_Od` or
    `block.seq4_Od`. All its statements but the last are ordinary statements at rung Cd, and
    may include loops, `if`s and other stores. The last is an Od statement on the way to the
    final store:
    - `stmt.if_Od` (its block an Od block);
    - `loop.*_Od` (its body an O(d+1) block);
    - `stmt.block_Od` (more than four);
    - `store.<buffer>_Od`, the final store itself.

    Read the signatures: `block.seq4_O0 : stmt.C0, stmt.C0, stmt.C0, stmt.O0 -> block.O0`.
  - **An epilogue block groups from the front,** unlike an ordinary block. It holds at most
    three C statements before its last. With more than three, the leading ones go into
    one `stmt.block_C<d>`, and the last two stay where they are. Statements c1 … c5, then the
    final O statement o, at depth 0:
    `(block.seq4_O0 (stmt.block_C0 (block.seq3_C0 c1 c2 c3)) c4 c5 o)`.
    (An ordinary block nests its *end*; see the next bullet.)
  - **Otherwise:** the body ends with the terminal `stmt.return_L0`, and the epilogue is
    `fallback`.
  - **An entry point that stores nothing** (only `return;`): its body is
    `(block.one_L0 stmt.return_L0)` and its epilogue `fallback`.
- **Blocks hold exactly 1, 2, 3 or 4 statements:** `block.one_<rung>`, `block.seq2_<rung>`,
  `block.seq3_<rung>`, `block.seq4_<rung>`, with exactly that many children. A sequence of
  more than four statements is written as its first three statements, then
  `(stmt.block_<rung> BLOCK)`, where BLOCK holds the rest by the same rule. Seven statements
  s1 … s7 at rung L1:
  `(block.seq4_L1 s1 s2 s3 (stmt.block_L1 (block.seq4_L1 s4 s5 s6 s7)))`; eight would be
  `(block.seq4_L1 s1 s2 s3 (stmt.block_L1 (block.seq4_L1 s4 s5 s6 (stmt.block_L1 (block.seq2_L1 s7 s8)))))`.
  A `{ … }` block inside the program is opened in place: its statements join the enclosing
  sequence.
- **Rungs.** A statement's type says where it stands, so loops and barriers stay legal by
  construction:
  - `L0`: top level. `L1`: inside the item loop. `L2`, `L3`, `L4`: inside one, two or three
    nested tile loops.
  - `C0` … `C4`: the same depths under a condition. No barrier is allowed there.
  - `O0` …: the epilogue.

  A row's name ends with the rung it produces (for example `reg.set_f_C1`). Never work a rung
  out yourself; read it from the table. Every row's signature gives the type of each child,
  including the rung of a loop's or `if`'s block (`loop.tilex_L1 : u32.boundu, block.L2 ->
  stmt.L1`: the loop stands at L1 and its body is a block at L2). Write each child at the
  type its parent's signature demands.
- **Loops** are rows whose counter is the loop's own (`safe.counter`, `safe.tile_counter`,
  `safe.tile_counter2`…). You never write a counter's increment; the loop row steps it. Loop
  bounds are typed bounds (`bound.*`, `uniform.*`, literal bounds), so every loop terminates.
- **Indices** are typed by meaning: an `index` is a position, a `count` is a length, `bits`
  is a bit field. They share the u32 slot but not the operators. Crossing between them uses
  only the `convert.*` rows. A store's index must be a row the table accepts for that buffer.
  The thread's own row and column rows (`index.of_row`, `index.of_col`, `safe.row_as_index`…)
  are how a thread names its own cells.

### A4. Terminals in play

Arity-0 symbols:

- **Literals.** A literal is `literal.<form>` with a value. The form fixes the type and the
  allowed range (the `literal` column of A5); a value outside it is a mistranslation. Forms
  include `real` (f32), `index`, `count`, `bits`, `int`, `flag`, and the bound and size forms
  `wgdim`, `blk`, `wglen`, `lit64`, `lit4096`.
- **Builtins:** `builtin.global_invocation_id`, `builtin.local_invocation_id`,
  `builtin.workgroup_id`, `builtin.num_workgroups`, read through the `axis.*` rows
  (`axis.x_lid` and so on).
- **The problem's bindings and uniform fields:** `load.<buffer>` rows (the buffer is the row's
  identity) and `load.cfg_<field>` / `uniform.cfg_<field>`. They are listed in A7.
- **Registers:** `reg.f0`… (f32), `reg.i0`… (u32 index), `reg.v0`… (vec4<f32>). Read one with
  `reg.get_*`, write one with `reg.set_*_<rung>`. Private arrays are `parr.a`, `parr.ua`…, and
  workgroup arrays are `wg.a`, `wg.ua`…, declared in the grid.
- **Counters:** `safe.counter`, `safe.tile_counter`, `safe.tile_counter2`, read through
  `safe.counter_index`, `safe.tile_index` and `safe.tile_index2`.
- **Literals in Karva** are one symbol with its value attached: `literal.index=512`,
  `literal.real=-3e38`, `literal.wgdim=16`.
- **`fallback`:** the empty slot, used for an entry point's missing epilogue (A3). Never in
  the padding.

### A4b. Reading the program's text

The programs you translate were written by this system's renderer from chromosomes, so
every construct in them is the rendering of one row. **The note column of the symbol table
(A6) shows the WGSL each row renders to:** match the program's text against those notes.
The renderer's habits:
- **Declarations are not statements.** The `var` declarations at the top of an entry point
  (registers `f0…`, `i0…`, `v0…`, the counters `t`, `k`, `k2`, the private arrays) are not
  statements; omit them. Name a register by its row (`reg.f0`, `reg.i3`).
- **Loops.** The test of a `for` decides its row:
  - **`t < min((X), 16777216u)`** is the item loop: `loop.items_*`, counter `t`
    (`safe.counter`), bound `bound.of_uniform` around X.
  - **`k < min((X), 16777216u)`** (or `k2`, `k3`) is a bounded tile loop of depth 1, 2, 3:
    `loop.tilex_*` (`safe.tile_counter`, `safe.tile_counter2`…), bound `bound.x_of_uniform`
    around X.
  - **X is a uniform value.** `lit_u32(N)` is `uniform.of_lit` with `literal.lit4096=N`, and a
    cfg field is `uniform.cfg_<field>`. The `min(…, 16777216u)` is what the bound row renders
    to, so it is not translated separately.
  - **Only a bare literal test, `k < 64u` (no `min`, no `lit_u32`),** is a `loop.tile_*` row,
    with `literal.lit64=64`.
  - A step other than `+ 1u` is a `_strided` row, its step a literal.
- **`lit_u32(N)`** is a literal; its form is the one the parent demands at that position.
- **Write a literal in the form demanded.** Where a `bits` value is demanded, write
  `literal.bits=65535`, not a conversion around a literal of another form. A `convert.*` row
  is only for a value the program converts.
- **Masks are rows:** `(x) & 63u` on a private-array index is `safe.pidx64`, and a mask on a
  workgroup-array index is that array's index row. Translate the masked value inside.
- **Order is meaning.** Keep each operation's operand order exactly as written.

Write the whole tree, however long. Never shorten, summarise or stop early.

### A5. The types (duals)

A type is a *dual*: a storage slot (`f32`, `u32`, `vec4<f32>`, `bool`…) paired with a meaning
(its form). The full list, with each one's meaning and literal range:

{{DUAL_TABLE}}
<!-- builder: kingdoms/safewgsl/types.tsv, columns dual, slot, form, meaning, literal;
     one line each: `T001 f32.real — a plain number — literal −3.4e38..3.4e38` -->

### A6. The symbol table

Every symbol you may use, by class. The format is `name : input types -> output type
[arity] — note`. A row whose types are written `<S.real>` stands for one instance per real
dual; the instance is the one whose types match where you use it. You must not use a name
that is not here.

{{SYMBOL_TABLE}}
<!-- builder: the problem's instantiated table, the rows the problem's interface can spell
     (fuller WgslKingdom + role_spells), grouped by class, one line a row:
     `arith.add : <S.real>, <S.real> -> <S.real> [2]` -->

### A7. The problem's interface

{{INTERFACE}}
<!-- builder: from the problem's contract.md and role: each binding (number, name, element
     type, read/write), the cfg fields, the entry-point names, and the rows each one gives
     (`load.keys : index -> vec4<f32>.real`, `load.cfg_n_queries : -> u32.count`, ...) -->

### A8. Worked examples

A whole kernel from another problem, with its own bindings. First its tree, written with
brackets only to show its shape; then its gene in Karva, the form you return.

{{FULL_EXAMPLE}}

### A9. The format you hand back

Begin your reply with the line ```karva. Reply with exactly two fenced blocks and nothing
else: no explanation and no working, before, between or after them.

1. **A block tagged `karva`:** the gene, one symbol a line, numbered from 0. The format is
   `<n> <symbol>`, and a literal's value is attached with `=` (`17 literal.index=512`).
   - After the last symbol of the ORF, write the line `ORF END`.
   - Then the right padding of A2b, numbered on: exactly 3 × F + 1 varied terminals, where F
     is the number of symbols of arity ≥ 1 in the ORF.
   - The root is `kernel.module_N` (or `kernel.entry_safe` for a single entry point).
   - Where nothing in the table expresses part of the program, write the symbol
     `untranslatable` there (arity 0), and say what in the notes.
2. **A block tagged `notes`:** one line per `untranslatable`, giving the program construct and
   the nearest symbols you considered; or `none`.

### A10. How your gene is checked

1. **Decoding.** It is read by A2b with the table's arities and types. The decoder reports:
   - every symbol that does not exist;
   - every position whose symbol's output type is not the demanded type;
   - every literal outside its range;
   - where the ORF ends, and whether that is at your `ORF END`.

   Every reply decodes to a valid gene, because the tail supplies any missing children.
2. **Running.** The expressed tree is rendered to WGSL and run against the program on the
   same inputs. Its outputs must equal the program's.

Reproduce what the program computes exactly, including the order of operations, since a
floating-point sum reordered is a different program.

---

## PART B — the per-call challenge

### B1. The challenge

Translate the program below into one safeWGSL gene in Karva (format A9), with its right padding. It is a kernel of
`{{PROBLEM}}` with entry points {{ENTRIES}}. Translate every entry point, in the order given.

```wgsl
{{PROGRAM}}
```

### B2. For a repair call (after a check failed)

Your previous gene is below, with the decoder's findings. Return the whole corrected gene in
format A9, changing only what the findings name.

```karva
{{PREVIOUS}}
```

```findings
{{FINDINGS}}
```

---

## Design notes for review (not sent to the model)

1. **The output is Karva** (Andrew, 2026-10-19). A brackets-based reply failed on bracket
   counting even from a top model (ChatGPT, version 2: 4 missing). Karva has no brackets, and
   with GEP's tail rule (3F + 1 terminals for K = 4) every reply decodes to a valid gene; a
   mistake is a mistranslation, found by decoding and by running.
2. **The padding is random valid terminals**, not `fallback` or no-ops (Andrew): if the
   model misjudges an arity, the decoder reads into the tail and still expresses a
   well-formed program.
3. **Prefix order as a second arm** (depth-first and bracket-free, also always decodable by
   arity) is still to measure against level order.
4. **Table size:** 1,768 template rows and 192 duals, about 55,000 tokens, the same in every
   call and so cached.
5. **Ground truth:** fuller's reader's tree for each program (`truth/*.safe.sexpr`). A reply
   is decoded and compared node by node by position in the tree, then run.
6. **Not in scope yet:** the model mutating a gene rather than translating one.
