# Prompt: translate a WGSL kernel into a safeWGSL GEP chromosome

Status: DRAFT for review (2026-10-19). Part A is the fixed header (the same for every call on
one kingdom and problem, so it is the cached prefix); Part B is the per-call challenge. Text
between `{{` `}}` is filled by the prompt builder from the kingdom's own tables, never typed
by hand. The design notes at the end are for us, not sent to the model.

---

## PART A — the fixed header (cached)

### A1. What you are doing

You are a translator for a gene expression programming (GEP) system that evolves GPU compute
kernels. You will be given a program and asked to write it as a GEP chromosome over a fixed,
typed symbol table. You do not write WGSL. You do not improve the program. You write the
genotype whose expression is this program, using only the symbols in the table below.

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

### A3. The kingdom: safeWGSL

The kingdom's symbols describe a whole compute kernel, statements and expressions alike:

- **The root** is `kernel.module_N` for a module of N entry points, each child a
  `kernel.entry_safe` (one entry point). An entry point is a grid (its workgroup size and the
  workgroup arrays it declares) and a body (a block of statements).
- **Statements** are rows too: assignments to registers (`reg.set_*`), private arrays
  (`parr.set_*`), workgroup arrays (`wg.put_*`), stores to output buffers (`store.*`), atomics,
  `if`, loops, barriers, return. Blocks (`block.seqN_*`, `block.one_*`) sequence 1–4
  statements; longer sequences nest.
- **Rungs.** A statement's type says where it stands, so loops and barriers stay legal by
  construction:
  - `L0`: top level. `L1`: inside the item loop. `L2`, `L3`, `L4`: inside one, two or three
    nested tile loops.
  - `C0` … `C4`: the same depths under a condition. No barrier is allowed there.
  - `O0` …: the epilogue.

  A row's name ends with the rung it produces (for example `reg.set_f_C1`). A block's rung
  must be the rung its parent demands.
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

- **Literals.** Write a literal as `(literal.<form> VALUE)`. The form fixes the type and the
  allowed range (the `literal` column of A6); a value outside it is a mistranslation. Forms
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
- **`fallback`:** the empty slot. Use it only where a parent demands a child you have nothing
  for (for example the third entry of a two-entry module).

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

### A8. A worked example

Program (WGSL):

```wgsl
if (wid.x < cfg.n_queries) {
    for (k = 0u; k < 512u; k = k + 1u) {
        f0 = f0 + dot(queries[wid.x * 512u + k], queries[wid.x * 512u + k]);
    }
}
```

Its chromosome:

```chromosome
(stmt.if_L0
  (compare.lt (index.of_col (axis.x_wid builtin.workgroup_id)) (index.of_count load.cfg_n_queries))
  (block.one_C0
    (loop.tilex_C0 (bound.x_of_uniform (uniform.of_lit (literal.lit4096 512)))
      (block.one_C1
        (reg.set_f_C1 reg.f0
          (arith.add (reg.get_f reg.f0)
            (geom.dot
              (load.queries (index.add (index.mul (index.of_col (axis.x_wid builtin.workgroup_id)) (literal.index 512)) (safe.tile_index safe.tile_counter)))
              (load.queries (index.add (index.mul (index.of_col (axis.x_wid builtin.workgroup_id)) (literal.index 512)) (safe.tile_index safe.tile_counter))))))))))
```

Points to notice:
- `wid.x` is a column position, `index.of_col (axis.x_wid builtin.workgroup_id)`.
- `cfg.n_queries` is a count, so it is compared as an index through `index.of_count`.
- The loop's counter is the tile loop's own, `safe.tile_counter`, and its body is at rung C1,
  because the loop stands under the `if`.
- The repeated `load.queries` subtree is written twice.

### A9. The format you hand back

Reply with exactly two fenced blocks and nothing else.

1. A block tagged `chromosome`, holding one s-expression, the whole kernel's tree:
   - `(name child1 child2 …)` for a symbol of arity ≥ 1, with exactly as many children as its
     arity;
   - `name` alone for an arity-0 symbol;
   - `(literal.<form> VALUE)` for a literal, with VALUE a decimal number (reals may use an
     exponent);
   - the root is `kernel.module_N` (or `kernel.entry_safe` for a single entry point).
2. A block tagged `notes`, one line per place where you could not find a row for what the
   program does. Each line gives the program construct, and the nearest symbols you
   considered. Write `(untranslatable "<construct>")` at that place in the tree, so the rest
   still checks. If everything translates, write `none`.

### A10. How your translation is checked

Every node is checked against the table:
- the name exists;
- the number of children is the row's arity;
- each child's output type is the input type its parent demands at that position;
- each literal's value is in its form's range;
- every statement sits at the rung its block demands.

Each failure is reported with its path in the tree (for example `root/1/2/0`). Then the tree
is expressed, rendered to WGSL and run against the program on the same inputs; its outputs
must equal the program's. Reproduce what the program computes exactly, including the order of
operations, since a floating-point sum reordered is a different program.

---

## PART B — the per-call challenge

### B1. The challenge

Translate the program below into a safeWGSL chromosome (format A9). It is a kernel of
`{{PROBLEM}}` with entry points {{ENTRIES}}. Translate every entry point, in the order given.

```wgsl
{{PROGRAM}}
```

### B2. For a repair call (after a check failed)

Your previous chromosome is below, with the checker's findings. Return the whole corrected
chromosome in format A9, changing only what the findings name.

```chromosome
{{PREVIOUS}}
```

```findings
{{FINDINGS}}
```

---

## Design notes for review (not sent to the model)

1. **Output format.** I chose a typed s-expression (the phenotype tree in prefix form), not
   Karva itself. Karva is breadth-first with head and tail lengths and homeotic hrefs, which is
   hard for a model to write correctly. The s-expression is the same tree; fuller's encoder
   (`safe_karva` and the homeotic fold) turns it into the Karva chromosome, with no loss.
   Alternative to test: ask for Karva directly, as a second arm.
2. **Table size.** The safe kingdom has 1,768 template rows and 192 duals. In the A6 format
   that is about 35,000 tokens, and the same for every call, so it is cached. Alternative arm:
   only the rows whose types occur in the program, about 300, which is cheaper but gives no
   way to find a better row.
3. **Ground truth.** fuller's reader already translates the v24 reference
   (`problems/v24_ingest_small/reference.wgsl`) into this exact form (its `.safe.sexpr`, 7,570
   characters), so each model's answer is scored node by node against the reader's, and then by
   running both.
4. **Scored per call:**
   - the share of nodes correct against the reader's tree;
   - the share of checks passed (A10);
   - whether the chromosome runs and equals the program;
   - seconds and tokens.

   The comparison is Haiku against Sonnet, on the v24 reference first, then on 3 of run 1's
   seeds.
5. **Not in scope yet:** the model *mutating* a chromosome rather than translating one. Once
   translation is accurate, the mutation call is the same header with "change this chromosome
   so that <instruction>" as Part B.
