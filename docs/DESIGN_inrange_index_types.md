# In-range indices as a type (wgsl_true)

**Objective.** A kernel the WGSL kingdom expresses must run as fast as its source text. Today every load and store of
a storage buffer spells the clamp `B[min(i, arrayLength(&B) - 1u)]`, because that is how the kingdom makes every index
legal by construction. Measured 2026-10-31 (`/private/tmp/claude-501/fbench/sus2.log`, `sus3.log`): the k-means scatter
form as expressed is 2.4-2.6 % slower than its source, and the clamps are the first cause.

**Approach.** "In range of buffer B" becomes a type, `u32.in_<B>`. A load or store of B takes that type and spells no
clamp: `B[{0}]`. Legality still comes from the types: only rows whose result is in range by their own text, or by the
interface's dispatch contract, give `u32.in_<B>`. The clamp is kept as one of those rows, so every index stays legal and
every gene still expresses a legal kernel, with no filter, repair or fallback.

**Status.** Built for the interfaces that declare `inrange_types` (`kmeans_update`, `kmeans_update_segmented`); the
other tables are unchanged byte for byte.

## 1. The types

| Type | Meaning | Contract it rests on |
|---|---|---|
| `u32.in_<B>` | a value `< arrayLength(&B)` | WebGPU forbids a zero-size binding, so `arrayLength(&B) >= 1` |
| `u32.len_<B>` | a value `<= arrayLength(&B)` | the interface's `lengths` (the problem's buffer sizes, in the uniform's fields) |

Both are `u32` slots, so the share pass binds them as it binds any `u32` value. Each has its `w` and `a` twins, as
every index type has.

## 2. The rows that mint `u32.in_<B>`

| Row | Spelling | Why the result is in range |
|---|---|---|
| `inrange.<B>_clamp(u32.index)` | `min({0}, arrayLength(&B) - 1u)` | the row's own text (the old clamp, now a row of its own: the fallback that keeps every index legal) |
| `own.u32_in_<B>` | `lit_u32(0u)` | `arrayLength(&B) >= 1` |
| `inrange.<B>_g<d>` (terminal) | `gi_<B><d>` | an invariant of the register: declared `var gi_<B><d>: u32 = 0u;`, and the ONLY text that assigns it is the guard row's, inside `if (gv < L)` with `L: u32.len_<B>` |
| `stmt.guard_lt_<B>_<rung>(u32.index, u32.len_<B>, block)` | `{ let gv = {0}; if (gv < {1}) { gi_<B><d> = gv; {2} } }` | mints the register's value: `gv < L <= arrayLength(&B)` |
| `stmt.guard_lt_else_<B>_<rung>` | `... if (gv < {1}) { gi_<B><d> = gv; {2} } else { {3} } }` | as above |
| `stmt.guard_ge_<B>_<rung>` | `{ let gv = {0}; if (gv >= {1}) { {2} } else { gi_<B><d> = gv; {3} } }` | the early-return form of an entry (`if (v >= n) { return; }`) |

The rows that give `u32.len_<B>`: `arrayLength(&B)`; each of the interface's `len_bounds` of B, a text over the uniform's
fields (`params.x` for `assignments`); and `len.<B>_of_in(u32.in_<B>)`, spelled `{0}` (`x < len` gives `x <= len`).

The guard rows exist at the L and C rungs of every depth and stratum (the body is a C block, as `stmt.if`'s); `<d>` is
the guard statement's depth, so a guard inside a loop body uses another register than one around the loop.

## 3. What the generator proves from the emitted text (the witness)

`gen_tables.py` refuses to write a table unless, for every row:

1. every subscript of a storage binding B in the spelling, `B[...]`, is `B[{i}]` with argument i of type `u32.in_<B>`,
   or the clamp text `B[min(..., arrayLength(&B) - 1u)]` (atomic rows and the vec4 helper keep it);
2. every row giving `u32.in_<B>` is one of the four forms of section 2, and every assignment `gi_<B><d> = ` in any
   spelling is a guard row's, inside its `if (gv < {1})` (or the `else` of `if (gv >= {1})`) with `{1}: u32.len_<B>`;
3. every `len_bounds` text of B is at most the interface's length of B for every parameter in the problem's
   `safety.contract` (a polynomial proof: each variable written as its lower bound plus a non-negative part, every
   coefficient of `length - bound` non-negative), and does not wrap at the contract's largest parameters.

The host side of the contract (the buffers really are that long) is a Rust test over every case of the problem.

## 4. Expression, mutation and crossover stay legal

The Karva walk asks each slot for a type and fills it from that type's reading frame. A load slot asks for
`u32.in_<B>`; whatever symbol fills it is one of the minting rows, so the index is in range whatever the gene. A guard
register read anywhere (outside its guard, before any guard ran, after a later guard) holds 0 or the last value a guard
assigned: in range in every case, only possibly stale, and fitness judges the value. Point mutation, transposition and
crossover draw each cell from its own type column, so they cannot place an `u32.index` where `u32.in_<B>` is asked.
Nothing is filtered or repaired.

## 5. The reader

The reader (`true_reader.rs`) reads `if (e < L) {A}` (and `if (e >= L) {A} else {B}`) as a guard row of buffer B when
`L` reads as `u32.len_<B>`, the guarded branch loads or stores B at an index equal to `e`, `e` reads nothing the branch
writes, and no guard of B at the same depth is open. Inside the branch, an index of B equal to `e` reads as the
register; any other index reads through the clamp row. So a source kernel keeps its own guards and loses only the
clamps its guards make redundant.

## 6. Not covered (yet)

An index that is in range only by the data (a slot read from scratch, a label read from `assignments`) or by affine
reasoning over a loop counter (`scratch[2 nb k + k + c]`, `c < k`) still reads through the clamp row. Offset types
(`base + c` with `base + C <= len`, `c < C`) are the next step if the measurement shows those clamps cost.
