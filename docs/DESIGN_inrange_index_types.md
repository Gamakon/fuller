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

## 6. Offset types (built 2026-10-31, `inrange.py` section 6)

A run of B is a region `[base, base + L)` the dispatch contract places inside B. `u32.upto_<L>` is a value at most the
uniform field L; `inrange.<B>_run_<r>(u32.upto_<L>)`, spelled `(<base> + {0})`, is in range because the witness proves
`base + L + 1` at most the contract's length of B. Its counter is a run loop's own register `kr_<L><d>`, which only run
loops assign.

## 7. Break guards (built 2026-10-31)

A loop that stops at a length, `let i = ..; if (i >= L) { break; } rest`, proves `i < L` for `rest`, as `if (i < L)
{ rest }` does. Measured: the clamp on reply 2's `assignments[base + q]` under such a break cost its hist entry 0.8 %
(0.0609 ms without the clamp against the source's 0.0616, `fbench/b4_hist.log`).

| Row | Spelling | Why the result is in range |
|---|---|---|
| `block.brk_guard_<B>_B<d>_f<k>(block.C, u32.index, u32.len_<B>, block.C)` | `{0} { let gv = {1}; if (gv >= {2}) { break; } gi_<B><d> = gv; } {3}` | the register is assigned only after the loop did not break, so `gv < L <= arrayLength(&B)`; `{3}` runs only after the assignment |
| `len.<B>_min(u32.index, u32.len_<B>)` | `min({0}, {1})` | at most `{1}`, which is at most the length (`end = min(base + 256u, params.x)`) |

The row is a loop body of the C rung (`block.B<d>`, as `block.brk_B<d>`), at depths 1-5, and its register is the one
of its blocks' depth, so the register stays an invariant of section 2: 0, or the last value a guard proved in range.

**Witness** (rules 2 and 3, extended): an assignment `gi_<B><d> = ` may also be the break guard's exact text, with
argument 2 of type `u32.len_<B>`, output `block.B<d>_f*` and both blocks `block.C<d>_f*`; a row giving `u32.len_<B>`
may also be `min({0}, {1})` with argument 1 of type `u32.len_<B>`.

**Reader.** The walk spells `if (c) { break; }` as `f = c; if (f) {}; break if f` with a register `f` of its own. When
`c` is `e >= L`, `L` reads as `u32.len_<B>`, the rest of the body indexes B at `e`, and the conditions of section 5 hold
(`e` reads a register or counter that the rest does not write, the rest holds plain statements, no guard of B is open
at this depth), the body reads as the break guard row; `f` is not read. Inside the rest, an index of B equal to `e`
reads as the register.

## 8. The pool's extent (built 2026-10-31, `pool.py`)

The interface's fixed 1,024-word workgroup pool cost reply 2's scatter entry 6 % against a 260-word pool (0.0252 ms
against the source's 0.0266, `fbench/b4_pool.log`): workgroup memory limits occupancy. An expressed kernel now
declares only the words its tree names.

| Piece | Spelling | Why every subscript stays in range |
|---|---|---|
| the extent `u32.wgext` | a constant of `4..=P` in steps of 4 (P the interface's `pool_words`), an argument of every entry, spelled `// pool {i} words` | the module declares the largest its entries name, E |
| the declaration | `const WG_PW: u32 = Eu;` and `var<workgroup> wg_pool: array<u32, E + 4>;` | written by `TrueKingdom::prologue_for` from the tree; the table's own lines hold P |
| every pool row and helper | the bound spelled `WG_PW`, never a number | a read is `min(.., WG_PW)` (or `xw`, which clamps so): at most E, below E + 4; a write is `if (.. && X < WG_PW ..) { wg_pool[X] = ..` |

The region rows' in-range proofs are those of the fixed pool with P replaced by `WG_PW`: they hold for every E,
because no row spells a bound as a number. Region bases (`m * T`) and split points are not capped by E; every access
through them is.

**Witness** (`pool.py`, gen_tables refuses a table that breaks it; `python3 pool.py DIR` on a committed one): the prologue
declares `WG_PW = P` and `P + 4` words; no text that subscripts the pool spells `Pu`; every subscript `wg_pool[X]` is
`min(.., WG_PW)`, `xw(..)` (the prologue's `xw` clamps onto `WG_PW`), or a write directly under `if (cond)` with
`X < WG_PW` a conjunct of `cond`. It found a hole of the fixed pool: the split phases' writes were guarded by the split
`sp` only, and `sp` is a region base, up to 511 blocks, past the pool; they are now guarded by `WG_PW` too.

**Reader.** An entry's extent is the words its placed regions take (`place`), rounded up to 4 and at least 4: reply 2's
one 256-word array gives `WG_PW = 256` and 260 declared words.

## 9. Injective store indices (design 2026-10-31, bug D; not built)

**The hole.** `store.scratch_*(u32.in_scratch, value)` stores any in-range cell from any invocation, so a gene can
express two invocations storing one cell: a nondeterministic kernel, caught today only by the bit checks on 8 cases.
The output's store is already its own cell (`u32.threadrow_*`/`u32.threadcol_*`); the scratch arena's is not.

**The type.** `u32.inj_<B>_<R>`: a cell of region R of B that no other invocation of the dispatch stores. A store to B
takes it. R is a region of the interface's layout, `[base_R, base_R + ext_R)`, with base and extent texts over the
uniform's fields, as the runs of section 6 are; the witness proves the regions of one layout pairwise disjoint and
inside the contract's length of B. One region per type is needed because two stores, each injective, are not injective
together: invocation `i` storing `scratch[gid.x]` and invocation `j` storing `scratch[nb * k + gid.x]` collide when
`j = i + nb * k`. Within one region, every store of an entry must use the same own map, so the minting rows below are
the region's only producers, and one entry's stores to one region compose only through one map.

| Row | Spelling | Why no two invocations store one cell |
|---|---|---|
| `inj.<B>_<R>_own(u32.own_<R>)` | `(base_R + {0})` | the own index is `gid.x`, below `ext_R` by its guard (`if (t >= ext) { return; }`, a guard row of section 2 over the region's extent): `gid.x` differs between invocations |
| `inj.<B>_<R>_strided(u32.index, u32.own_k)` | `(base_R + {0} * k + {1})` | `{1}` is an own index below `k` (the stride, a uniform field): two invocations have different `{1}`, so different residues mod `k`; `{0}` is a loop counter below `ext_R / k` |
| `inj.<B>_<R>_run(u32.own_<R>)` | `(<run base> + {0})` | a run of section 6 plus an own index below the run's length |
| `inj.<B>_solo(u32.index)` | `{0}` inside `wg.solo*` of an entry the dispatch contract runs as ONE workgroup | one invocation stores at all (reply 2's `starts`: `lidx != 0u`, one workgroup) |

`u32.own_<R>` and `u32.own_k` are minted only by `gid.x` (or `wid.x * WG + lidx`, reply 2's scatter) under a guard
row whose length is `ext_R` (or `k`). Point mutation, transposition and crossover draw from the type's column, so an
`u32.index` never fills an injective slot, and a guard register read outside its guard holds 0 or a value a guard of
the same region proved: still the invocation's own value, since only own maps assign it.

**The data-indexed store has no such proof.** The scatter's `scratch[ids + starts[a] + prefix[b][a] + rank]` (segmented
line 84, reply 2 line 90, the champion's line 224) is injective only if the scratch holds the counting sort's true
counts and scans, which other entries' genes wrote. A gene whose `hist` body is `{}` (legal: the scratch is zero,
problem.toml `fill = "zero"`) leaves `starts = prefix = 0`, so the slot is the rank, and the first point of every
block stores cell `ids + 0`. No type of the scatter's own gene can see that, so "offsets[bin] + rank" cannot be a
minting row. The true-GEP plan says the same (`PLAN_true_gep_wgsl.md`, the rule ledger's layer 7: "scatter writes go
through atomics"). The construction that is deterministic for ANY slot value is `atomicMax(&ids[slot], v + 1u)` into a
zero region: the result is order-independent, and equals the plain store (plus one) whenever the slots are distinct,
as they are for a correct sort. It needs the ids region as an `array<atomic<u32>>` binding of its own, an interface
change to UltraDim's driver, and its cost must be measured against the plain store before it is adopted.

## 10. Not covered (yet)

An index that is in range only by the data (a slot read from scratch, a label read from `assignments`) still reads
through the clamp row.
