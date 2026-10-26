# wgsl_true: the true multi-typed WGSL kingdom

**Every gene expresses a legal GPU program, because the symbol table's types describe exactly the legal programs.**

There is no repair, filter, substitution or refusal anywhere. The renderer only substitutes argument texts into each row's spelling, and fitness (HFF) judges every kernel.

- **Design:** `nucleotable/docs/MultiTypedGEP.md` rev 4, rendered as `docs/MultiTypedGEP.html`.
- **The WGSL section:** `nucleotable/docs/MultiTypedGEP_s9_WGSL_draft.md`, batches 1–9.
- **Goal:** `docs/GOAL.md`.
- **Plan:** `docs/PLAN_true_gep_wgsl.md`.

## How a kernel is a gene

A kernel is one Karva gene with a head and a tail for every type: Ferreira's layout, one reading frame per type. A slot that asks for type T reads the next symbol of T's frame. Each legality rule of WGSL is carried by a type, so a symbol exists only where its use is legal:

| Rule | Carried by |
|---|---|
| no two threads write one element | ownership types (`u32.threadrow_<b>`, `u32.wgown_<j>`, owned pixels, slot, pair, block and segment phases) |
| barriers only in uniform control | statement rungs: `L` (uniform), `C` (conditional, no barrier), `W` (write phase between barriers), `O` (the epilogue, which always ends in the primary output's store) |
| every loop ends | bounded loop types: literal or capped trip counts, and fuel strata `_f<k>` up to the interface's budget K |
| every index is in range | index types masked or clamped by the row's own text |
| no undefined behaviour | WGSL-defined results only. Floating-point edge values are values, and fitness scores them |

## The files

| File | What it is |
|---|---|
| `gen_tables.py` | the generator: writes one interface instance's tables from the family modules |
| `interface.py` | the interface instances: bindings, uniforms, fuel budget, outputs |
| `skeleton.py`, `values.py`, `regs.py`, `loops.py`, `wg.py`, `vectors.py`, `arena.py`, `pool.py`, `image.py`, `lib.py` | the row families, one module per part of the language: kernel skeleton, values and arithmetic, registers and private arrays, loops and counters, workgroup memory and phases, vectors and matrices, storage arenas and atomics, the workgroup pool and its phases, texture output |
| `gen_share.py` → `share.tsv`, `share_constants.tsv` | the rows for shared code: a `let` within one kernel |
| `types.tsv`, `functions.tsv`, `constants.tsv`, `interface.tsv` | the generated tables. `functions.tsv` holds one row per symbol: output type, ordered argument types, WGSL spelling |
| `prologue.wgsl`, `declarations.wgsl` | the interface's prologue (bindings, helpers) and the declarations each row brings |
| `img/<shader>/` | image-output instances (the cuneus visual shaders) |
| `mod/<name>/` | two-entry module instances (UltraDim ingest and search under one fixed linker) |

Generate a table:

```
python3 gen_tables.py                    # the Gram instance, beside this file
python3 gen_tables.py img_volumepassage  # into img/volumepassage/
python3 gen_tables.py v24                # into mod/v24/
```

## Instances (2026-10-23)

| Instance | Types | Rows | Used for |
|---|---:|---:|---|
| `gram` (default) | 305 | 8,841 | the first unit test: a weighted Gram matrix |
| `img/volumepassage` (K = 2) | 361 | 10,534 | the cuneus shader volumepassage, read exactly |
| `img/volumepassage_k1` | 305 | 8,087 | the same at K = 1 |
| `img/nebula` | 361 | 10,564 | the cuneus shader nebula, read exactly |
| `mod/v24` | 1,369 | 77,524 | UltraDim GPU ingest + search |
| `mod/v24_k1` | 559 | 25,824 | the same at K = 1 |

## Where it stands (measured)

- **Legality:**
  - 1,000 of 1,000 random genes compile (naga and the Metal pipeline) and run, per table version;
  - 10,000 random two-entry modules pass naga;
  - every rewrite of the in-loop optimisers keeps its output store (10,000 tested).
- **Reading real code:**
  - the cuneus shaders volumepassage and nebula draw pictures identical to the originals;
  - 50 of 67 UltraDim ingest kernels read exactly, 14 are named outside the kingdom (races in their own text), 3 are gaps.
- **Speed parity:** the generated volumepassage runs at 0.983 of the shipped shader's time.
- **Evolution** (phylu `true_wgsl_fit`):
  - planted slowdowns are recovered: volumepassage 1.74× → 0.96× shipped, and the ingest seed27 repeat from 4 to 1;
  - no faster volumepassage and no better UltraDim ingest yet. Reports are in `docs/reports/2026-10-22_ST9_*` and `docs/reports/2026-10-23_*`.

## History

- **2026-10-21:** generated from the design (bf0d074b). The first compiler loop found its row defects, each fixed in the table.
- **2026-10-21 to 22:**
  - the Apple Metal loop hang was found and the loops respelled (7f1b59c0);
  - image output (737441fc);
  - share-equivalence (b9077185);
  - the in-loop optimisers (326326ff);
  - speed parity with shipped code (37f19dbc);
  - the epilogue store fix (ba8e7fd1).
- **2026-10-22 to 23:** the UltraDim two-entry module and its phases (e7a881b4 … 66d2ce94).

`git log -- kingdoms/wgsl_true` lists all 44 commits.

## Open

- The table-as-grammar text parser (plan UI5).
- A work budget per workgroup carried in the types (the fuel-counter proposal in the s9 draft).
- The three remaining ingest-kernel gaps.
