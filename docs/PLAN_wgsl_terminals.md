# PLAN: a real terminal for every safeWGSL dual

## Context

**Objective.** Make the live WGSL kingdom (`kingdoms/safewgsl`, loaded by `WgslKingdom::load_safe`) pass the multi-typed GEP kingdom check (`src/kingdom_contract.rs`): every type a gene root or a function argument can demand has at least one terminal row, so every tail can be filled with a symbol of its own type and no fallback exists anywhere.

**Approach.** Each dual that has no terminal gets a real, named terminal row, added in `kingdoms/safewgsl/gen_tables.py` (never by hand in the TSV) and written as real WGSL text. Nothing is substituted at decode time; a terminal is an ordinary symbol that selection ranks like any other.

**Unique feature.** A tail cell of type T holds a terminal of T, so a dual with no terminal is an unfillable tail. The check reports it as "our software has a bug: type X is demanded and has no terminal".

**Status.** Plan, nothing added. Measured 2026-10-19 on `WgslKingdom::load_safe()`: 204 duals, 3,352 rows. 75 duals have a terminal; 129 do not. The older `kingdoms/wgsl` (83 duals, 59 without) is superseded by this plan.

**Why.** Without these rows the live kingdom cannot host a typed-GEP gene (CR section 9 item 3).

## The 129 by group

| Group | Duals | Count | Status |
|---|---|---|---|
| A. value duals with a literal domain | T010-12, T018-20, T022, T024, T030-32, T035-37, T061-69 | 23 | approved: `?_T` |
| B. value duals with no literal domain | T013-16, T025-28, T033, T038, T039-58, T059, T060 | 32 | approved option 1: give a domain and a `?_T` |
| C. statement and block levels | see below | 41 | approved: a named empty block |
| E. provenance, index and bound duals | see below | 33 | needs a ruling |

23 + 32 + 41 + 33 = 129.

### A. 23 value duals: `literal.<form>` over the dual's existing domain

The scalar forms already have `literal.<form>` rows (F001-F006); the vector and matrix duals get their own row, rendered by splatting one `C_T` value over the lanes, as the existing literal rows already do. Domains are the `LITERAL_DOMAIN` entries in `gen_tables.py` for `int`, `index`, `count`, `bits`, `flag` and `real`.

### B. 32 value duals: add a `LITERAL_DOMAIN` entry and a `literal.<form>` row

| Form | Duals | Domain |
|---|---|---|
| `code4`, `code8`, `code16`, `code1024` | T025-28 | `0..=3`, `0..=7`, `0..=15`, `0..=1023` |
| `sign` | T038 | `0,1` (false, true) |
| `q15_16` | T013-16 | any i32, `-16777216..=16777216` (the f32-exact range every integer literal uses) |
| `q15_16w` | T059 | the same, per lane |
| `hash32`, `hash64` | T033, T060 | `0..=16777216` per lane (see the limit below) |
| `opaque` | T039-58 | the scalar's literal domain; f16 has none (the renderer writes no f16 literal) |

Two limits I found, for you to rule on:
1. **f16 opaque (T043-46, 4 duals).** `literal_domain()` returns `-` for f16 because the renderer writes no f16 literal. A real `?_T` needs the renderer to write one. I propose emitting `f16(<f32 value>)`, which is real WGSL when the `f16` extension is on (the module prologue already enables it for f16 kernels).
2. **Full-range hash and opaque u32/i32.** A literal's `konst` is an f32, exact only to 2^24, and the safe reader refuses larger literals (see the comment above `LITERAL_DOMAIN`). "The full range of the scalar" therefore cannot be a single literal row without a reader change. I propose the 2^24 range for now and a second step (halves, `(hi << 16u) | lo`, which the reader already knows) if you want the full 32 bits.

### C. 41 statement-level duals: a named empty block

| Duals | Count | Row (text) |
|---|---|---|
| T074 `block.block` | 1 | `block.empty` `{}` |
| T097, T099, T101, T116, T121, T130 `block.L0..L5` | 6 | `block.empty_L<n>` `{}` |
| T102-T107, T117, T118, T122, T123, T131, T132 `stmt.C0..C5` and `block.C0..C5` | 12 | `stmt.empty_C<n>`, `block.empty_C<n>` `{}` |
| T133-T142 `stmt.O0..O4`, `block.O0..O4` | 10 | see the note below |
| T163-T167 `block.B1..B5` | 5 | `block.empty_B<n>` `{}` |
| T071 `store.store` | 1 | `store.empty` `{}` |
| T072 `kernel.kernel`, T108 `kernel.safe`, T187 `kernel.module` | 3 | `kernel.empty`, `kernel.empty_safe`, `kernel.empty_module`: an entry point with an empty body |
| T157 `wg.decl`, T158 `wg.decls` | 2 | `wg.decl_none`, `wg.decls_none`: no declaration (empty text) |
| T073 `grid.grid` | 1 | `grid.unit` `@workgroup_size(1, 1, 1)` |

Note on the epilogue (O0-O4): an O statement or block must END IN THE STORE of the thread's cell (the 2026-10-11 rule: a gene that cannot write h is a kingdom bug). An empty `{}` breaks that rule, so for O the terminal is a store of the thread's own cell, `h[cell] = h[cell] + f0` at the thread's first cell, as a named row `stmt.store_own_O<n>`, `block.store_own_O<n>`. This is the old fallback's text, but as an ordinary row, selected or not, and not a substitution.

Likewise `kernel.safe` and `kernel.module` bodies: an empty body fails the same rule, so their terminals are a kernel whose epilogue is that store, not `{}`. Please confirm.

## E. 33 provenance, index and bound duals: needs a ruling

These forms exist so that a bad program has no spelling. A constant would break the guarantee the form carries (a literal is not "the cell this thread owns"), so group E cannot take `?_T`. For each I propose the natural real value the form already names, as a terminal row:

| Duals | Form meaning | Proposed terminal |
|---|---|---|
| T090 `threadrow`, T091 `threadcol` | row/col from the thread's ids and block factor | `safe.own_row`, `safe.own_col`: the thread's first row/column |
| T149-152 `threadrows/cols/rowg/colg` | strided and grouped variants | `safe.own_row_s` etc., one per kind |
| T179-184 `rowbase`, `colbase` and `s`/`g` | the origin of a thread's row/col | `safe.origin_row`, `safe.origin_col` etc. |
| T092 `cell` | the output cell this thread owns | `safe.own_cell` |
| T185 `mirror` | the mirror of an owned cell | `safe.own_mirror` |
| T094 `wgindex`, T126 `wgvindex`, T155 `wgv2index`, T161 `wgaindex`, T111 `pindex`, T173 `pindex16`, T195 `pindex128` | an index masked into its array by its row | `literal.<form>`: `0..=N-1` of the array (a literal is safe, since the consumer masks it) |
| T186 `wgown` | an element this thread owns | `safe.own_element` (`lidx`) |
| T146 `rowid`, T147 `colid` | the y/x of a thread id | `axis.y_gid`-style rows take a `gid` terminal already; propose `axis.y_own`, `axis.x_own` |
| T127 `boundx`, T175 `boundu`, T176 `uniformw` | loop bounds, workgroup-uniform values | `literal.<form>` over `1..=4096`, uniform by being a literal |
| T188 `uflag` | a uniform predicate | `literal.uflag` `false, true` |
| T093 `phi`, T095 `bound`, T178 `indexw`, T192 `payload`, T070 `array<T>` | consumed by no row | root-only or unused; no terminal needed if they are not roots |

These "own" rows are what the old fallback table named ("the thread's own first row"). The difference is that they are ordinary rows a gene selects, not a decode-time substitution. If you prefer these 33 to be unreachable instead, the only alternative is to remove their demand from the signatures, which changes the kingdom's safety proofs; I have not proposed that.

## Roots

I checked the readers: the head gene of every program is rooted at `kernel.safe` (`check.rs:540`, `safe_karva.rs:690`), or `kernel.module` for several entry points (`safe_karva.rs:690`, `params.rs:208`). The other genes of a chromosome are homeotic definitions, rooted at whatever dual they define, and are referenced through an href terminal of that dual. `kernel.kernel` is a root only in an older gram test (`gram.rs:2074`).

So the roots for the check are `kernel.safe` and `kernel.module`. This removes the root-only demand on T070, T071, T072, T093, T095, T178, T192 (7 duals) from the check, but 5 of them still appear in the 129 and keep the rows above if you want the kingdom to hold them; I propose to give rows only to the demanded ones (the 122 with consumers) and list the 7 as "not demanded".

## Order of work

1. Rulings on B limits (f16, 2^24), the O/kernel body, and group E.
2. gen_tables.py rows for A, B, C; regenerate `types.tsv`, `functions.tsv`, `TYPES.md`, `symbols.md`; run the reader and renderer tests, which may need rendering code for each new row (`src/wgsl/render.rs`).
3. Re-run the kingdom check on safewgsl and on the older kingdom; closure on all duals (no subset) at 100,000 genes.
4. Group E per ruling.
5. Remove the `covered()` subset helper from `src/typed_gep_tests.rs`.

## Built (2026-10-20)

Rulings of the lead applied; both `kingdoms/safewgsl` (204 duals, roots `kernel.safe` and `kernel.module`) and `kingdoms/wgsl` (83 duals, root `kernel.kernel`) pass the kingdom check. The rows are generated by `gen_tables.py`, never by hand.

| Rows | Class | Duals | Text |
|---|---|---|---|
| constants of the duals no literal reaches | `const.<form>` (`const.vint`, `const.q15_16`, `const.opaque`, `const.mat`, ...) | value duals | the literal text of the dual's form, over the whole range of its scalar; an f16 constant is `f16(1.5f)` and the module enables f16 |
| provenance and bounds | `own.<slot>_<form>` | thread row, column, cell, origins, masked indices, bounds, uniform values | the thread's own row, column or cell, a thread id's axis, or index 0 / bound 1 |
| statements and blocks | `own.stmt_<rung>`, `own.block_<rung>`, `own.block_block` | C, B, L and O levels | an empty block, and for O the store of the thread's own cell to the interface's output binding with the row's constant |
| kernels and grid | `own.kernel_safe`, `own.kernel_module`, `own.kernel_kernel`, `own.grid_grid`, `own.wg_decl`, `own.wg_decls` | roots | the default grid, an empty body and the epilogue's store |

Not done: the 7 duals with no consumer get no row (`payload`, `bound`, `phi`, `indexw`, `array<T>`, `store.store`, `kernel.kernel` in safewgsl). A constant is an f64 (exact for every u32 and i32); phylu's `C_T` holds the bit pattern and converts at the boundary.
