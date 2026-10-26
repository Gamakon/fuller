# Assessment: first-class shapes in the multi-typed GEP

**Subject:** `docs/FirstClassShapes.md` (the proposal), read against the built engine.
**Date:** 2026-10-26 · **Branch:** `feat/futhark-kingdom` · **Status:** assessment only — nothing implemented.
**Reviewed:** adversarially, by `codex exec` on the command line; the review and its disposition are in `docs/reviews/2026-10-26_first_class_shapes_codex.md`.

---

## 0. The five answers, up front

| # | §9 question | answer |
|---|---|---|
| 1 | Is the proposed approach feasible? | **Yes, but not in the proposed form.** Symbolic *unification at expression time* (option A) cannot work under this design and would break the invariant it is meant to protect. The same expressive power is reachable **for a fixed, declared finite interface** if the unification is done **once, offline, by the kingdom generator**, and the result is a finite table of *shape-specialised* rows. |
| 2 | Which invariants / assumptions are affected? | The GEP engine's core assumptions are **not** affected. What changes: the *meaning of a type* (base type × shape class), the kingdom's tables (generated per interface), the Futhark rendering (shape-polymorphic source), the e-graph's datatype, rules **and bridge** (all generated), and the interface contract (a first-class artefact). §7. |
| 3 | Which option is most promising? | **A hybrid of B and C, resolved offline**: option C's *shape-safe primitives* (matmul, transpose, matched-shape zip) instantiated over a *declared, finite set of shape classes* (option B), with the class set emitted by the generator. **Option A is not viable.** |
| 4 | Can the fixed reading-frame design be retained? | **Yes — the core engine needs no change**, provided the shape class becomes part of the *type key* and a layout is immutably bound to one versioned kingdom/interface (§7). The frame, the FIFO walk, the alphabet rule, the bound table, `validate`, `place`, `fits`, `write_back` all stay as they are. |
| 5 | What is the minimum PoC? | A **class-specialised** PoC kingdom (not the parametric one the proposal sketches), with the seven experiments re-specified in §10. |

**Legality by construction is preservable.** The proposal's central fear — that shape constraints cannot live in a fixed Karva reading frame — is correct *about parametric frames* and wrong *about the frame mechanism*. A frame is a per-type sequence of interchangeable cells; if a shape class is a type, every cell of that frame has a static required class and a static alphabet, exactly as today.

**The one thing the proposal gets wrong** is the assumption that the shape relation must be checked at expression time. It must be *resolved* before the gene exists, and the place to resolve it is the table that already generates the kingdom.

**Three limits the proposal does not state, and this assessment now does** (§6.4, §12):

- the design is **quadratic in dimension roles for the genome width and cubic for the row count**, so it has a practical ceiling of roughly **3–5 roles**;
- the **generator becomes the new trust boundary** — the guarantee moves from the engine to the table, so the table needs its own independent checking;
- **runtime-equality programs** (two inputs whose sizes are equal at run time but unrelated in the type) are *out of scope*, and that is a real loss against the current 1-D kingdom.

**And the premise needs correcting** (§2): today there are no compile errors from shape, and no shapes to violate. The measured failure in this kingdom is a **compiler hang** in Futhark's optimiser, reproduced on three compilers.

---

## 1. What was inspected

Read in full, not skimmed:

| artefact | what it fixes |
|---|---|
| `src/geneframe.rs` | `Ty` (a finite enumerated set; `Ty::Futhark(u16)` is a *table-defined family* — one variant for the whole family, index = row of `types.tsv`), `SymbolTable`, `Ty::code`/`from_code`, the dual-count guard. |
| `src/kingdom_contract.rs` | `KingdomRelations` → `check()` → `KingdomContract`: `symbol_outputs` (exactly one output type per symbol), `argument_signature` (one required type per `(symbol, arg_index)`), and the derived `a_{T,U}`, `a_T`, `n_U`, `reachable`, `heads_from_program_size`, `bounds` (§2.1–2.2 of the design). |
| `src/typed_gep.rs` | `Frame`, `Span`, `GeneLayout`, `alphabet`, `mutate`, `swap_range`, `invert`, `validate`, `express` (the FIFO walk), `required_heads`, `place`, `fits`, `write_back`, `promotions`. |
| `src/futhark_kingdom.rs` | the loader: `FutharkDual`, `FutharkRow`, `relations()`, `render()`, `const_len()`, the four-type / 38-row assertions. |
| `kingdoms/futhark_true/{types.tsv,functions.tsv,gen_tables.py}` | the generator's output and its shape: 4 duals, 38 rows, `ref/name/kind/out/args/spelling/note`. |
| `src/futhark_egraph.rs` | the e-graph: one egglog sort per kingdom type (`FR`/`FA`/`FS`/`FF`), 38 constructors + `FId`/`FCompose`, `import`, `render`, `to_tree`, `saturate`, `heritable`. |
| `nucleotable/docs/MultiTypedGEP.md` | the design the engine implements, §1 (contract), §2 (layout and bounds), §3 (expression), §5 (variation: *"every operator draws from a domain enumerated from the layout before the draw"*), §6 (epoch alphabet). |

---

## 2. The premise, checked against the evidence

The proposal and the accompanying message both rest on: *"we are having compile errors because our kingdom is not designed on the full language."* What the measurements in this repository show is different, and it matters for how the shapes work should be prioritised.

**There are no compile errors.** The kingdom's own test compiles 12 random programs; the sweep behind `docs/ASSESSMENT_futhark_kingdom.md` compiled 144 and got 143 agreements, the one disagreement being `f32.round`'s semantics (a *value* difference, documented in §10.5 of that assessment). The Phase-3 rescue found something else entirely: the original gene and **all eight** e-graph candidates **timed out** at ~120 s (`docs/RESCUE_futhark.md` §4). A timeout is a hang, not a rejection. The hang is a Futhark optimiser defect, reproduced on 0.28.1, on upstream master 0.29.0-nightly, and on the `webgpu` branch at `14e9ccbf` (`docs/RESCUE_futhark.md` §6.1).

**There are no shapes to violate.** The kingdom is one-dimensional by construction: `arr.f32` is *"a one-dimensional array of reals"*, `scn.f32` is *"a one-dimensional array of reals that IS a scan result"*. There is no dimension in any type, no dimension in any row, and no row that creates a *competing* array length. The array operators are `map`, `map2`, `reverse` and `scan`, all of which preserve the length; `reduce` is the one rank-changing row — it consumes an array and yields a scalar, so it eliminates the length rather than producing an incompatible one. **An internal shape incompatibility cannot be expressed, so it cannot be compiled wrong.** The proposal's own rule — *"a compiler rejection caused by an incompatible internal shape must be treated as a failure of the proposed genetic design"* — cannot fire against the current kingdom; it is a rule for the *post-shapes* kingdom.

**What the kingdom does have is input-interface contracts, and they already behave correctly.** `map2 (+) xs ys` needs equal lengths. Measured on Futhark 0.28.1 (transcript: `/tmp/shape/futhark_shape_transcript.txt`):

```
map2 (+) [1.0,2.0] [1.0,2.0,3.0]   →  COMPILE ERROR
    Error at map2_lit.fut:1:41-54:
    Cannot apply "map2" to "[1.0, 2.0, 3.0]" (invalid type).
    Expected: [2]f32    Actual: [3]f32
    Sizes "2" and "3" do not match.

map2 (+) xs ys  (xs, ys dynamic)  →  COMPILES; runtime error on a mismatch:
    ./map2_dyn: Error: entry point arguments have invalid sizes.
```

So Futhark *does* catch a statically-known internal size mismatch at compile time, and falls back to a runtime obligation when the sizes are dynamic. That is precisely the distinction the proposal draws in §6 — *"shape compatibility guaranteed by the program"* versus *"shape requirements placed on external inputs"* — and the current kingdom sits entirely on the second side. **The current kingdom is already sound with respect to shapes.** What it lacks is the ability to *state* a shape relation in the type, and therefore to have Futhark's first behaviour (compile-time rejection) apply to internal relations too.

**Consequence for the plan.** The shapes work is a *capability* addition (express new program shapes), not a *bug fix* (stop illegal programs). It should not be scheduled as a repair of a broken kingdom. The one thing that would justify urgency is a *compiler rejection* that names a shape; none exists in this repo's measurements, and the hang is a separate, already-diagnosed Futhark defect.

---

## 3. The engine's legality argument, stated exactly

The guarantee rests on two properties of the built machinery. Both are worth naming precisely, because everything in §4 follows from them.

**(U1) Cell uniformity.** For each type `T`, `head_T` and `tail_T` are *homogeneous bags*: every cell of `head_T` holds a symbol whose output is `T`, and every cell of `tail_T` holds a terminal of `T`. `GeneLayout::alphabet` (`src/typed_gep.rs:223`) returns **the same set for every cell of the frame** — for a head cell, all `s` with `k.output(s) == t`; for a tail cell, `k.terminals_of(t)`. `validate` (`:524`) enforces the same equality per cell, read or not read.

**(U2) Slot determinacy.** A **non-root** slot's type is determined by the pair *(parent symbol, argument index)*: `express` pushes `k.signature(symbol)[i]` as the child's slot type (`:597`). The root slot's type is the layout's root. Nothing about a slot depends on its position, its ancestors, or anything else.

The design adds one more property that the argument depends on, and it is stated in the design itself (`nucleotable/docs/MultiTypedGEP.md` §5): **an edit in `head_T` changes only `T`'s bytes; it may change which rows other frames express, never their validity.** That is what makes mutation and crossover *domain-local*.

From (U1), (U2) and the locality property, legality by construction follows in one line: a slot of type `T` reads a cell of `head_T`/`tail_T`, whose contents are by (U1) a symbol of output `T`; by (U2) its children have that symbol's signature types; by induction the whole tree is well typed, and `place` (`:629`) re-checks exactly this on the way back.

The bound table is keyed the same way: `a_{T,U}`, `a_T`, `n_U`, `reachable`, `P_T`, `L`, `t_T`, `d_T` are all maps keyed by `Ty` (`src/kingdom_contract.rs:60-70`), and the *width* of a gene is the sum over **types** of `(h_T + t_T + d_T)`.

**This is also why the current kingdom has four types and not more.** `Ty::Futhark(u16)` is a table-defined family: one variant for the whole family, with the row index as the payload (`src/geneframe.rs:115`). A new type is a **table row**, not a code change — that mechanism is exactly what a shape class would use.

---

## 4. What symbolic shapes break

Take the proposal's own `matmul`:

```
matmul : Matrix(T,n,m) × Matrix(T,m,p) → Matrix(T,n,p)
```

Suppose shape classes are *not* part of the type — the type stays `Matrix`. Then in a frame keyed on `Matrix`:

- **The root slot** requires `Matrix(T,n,p)`.
- **Its first argument slot** requires `Matrix(T,n,m)` *if the root symbol is `matmul`*, but `Matrix(T,p,n)` *if the root is `transpose`* — because a transpose producing `Matrix(T,n,p)` must take `Matrix(T,p,n)` (`transpose : Matrix(T,p,n) → Matrix(T,n,p)`).

So the required class of the second `Matrix` cell depends on **which symbol the first cell holds**, which is exactly what (U2) forbids being dynamic. Three consequences follow, and they are the whole argument:

1. **The alphabet becomes path-dependent.** The set of symbols legal in a `Matrix` cell would depend on the binding context θ established by the ancestors. `alphabet` must be enumerable *before* the draw (`MultiTypedGEP.md` §5: *"every operator draws from a domain enumerated from the layout before the draw, so every child is valid by construction"*). With θ dynamic, it cannot be.

2. **Over-approximation breaks the invariant.** The alternative — enumerate the union over all θ — admits symbols that do not unify with the actual context. The gene then expresses a shape-ill-formed tree, and the only ways out are rejection, repair or substitution. All three are forbidden by the proposal's §1 and by the engine's design (`LayoutDefect` is *"our software has a bug"*, never a repair).

3. **Crossover loses locality.** `swap_range` exchanges a range inside one domain of one type and requires homologous layouts. With a path-dependent alphabet, a cell's legal contents differ between two genes, so exchanging cells can produce an illegal gene — and it would break the design's own statement that an edit "may change which rows other frames express, never their validity". `homologous` (`:176`) compares `head_lens()` maps keyed by type, which would no longer capture the shape binding.

**And the binding itself cannot be carried in the gene**, because the gene's words are symbol ids and constant indices — there is nowhere to put `θ`, and adding a word per cell would change the Karva representation that the design fixes as canonical.

So: **a parametric frame is not merely hard; it is inconsistent with this design's requirement that the domain be enumerable before the draw and that an edit never change another frame's validity.** That requirement is the mechanism by which "no repair, no rejection" is achieved. Option A asks the frame to give up exactly that.

---

## 5. Option A, stated as a lemma

**Lemma.** *No reading-frame mechanism compatible with this design — a fixed per-domain alphabet derived from `(layout, position)`, edits confined to one domain, and the guarantee that an edit never changes another frame's validity — can have a path-dependent cell alphabet and preserve legality by construction.*

*Proof.* Legality by construction requires that the legal values of a cell be computable without executing the walk; otherwise either the draw is illegal, or the value is validated after the fact. The design's locality property additionally requires that an edit in one domain cannot change another domain's validity, which is only true if a cell's legality is a function of its own domain alone. Under parametric shapes, the legality of a cell's value depends on the substitution θ accumulated along the path from the root to that cell, and θ is a function of the symbols already chosen — i.e. of the walk. ∎

**What the lemma does not say.** A *sequential* decoder is conceivable: re-derive the domain at each step from the prefix chosen so far, and never reject or repair. That is not this design — it gives up the fixed layout (a cell's value would depend on its prefix), the domain-locality of edits, and the crossover model (ranges could no longer be exchanged blindly). It is option A in all but name, and the proposal's §1 forbids the repair machinery it would require. The lemma is about the design as it stands.

**The lemma does not kill the idea of shapes. It kills the idea of *resolving shapes during expression*.** There are exactly two ways to satisfy it:

- **(A1) resolve statically** — θ is known before the walk. But θ at a cell is a function of the *parent's instantiation*, and the parent's instantiation is a function of *its* parent's θ. For θ to be static, the parent symbol's required class must be determined without knowing θ — which forces the class to be a function of the symbol id alone, i.e. **the class must be part of the type**. That is option B.
- **(A2) carry the shape as data** — a shape value threaded through the program, resolved at runtime. This is not expressible in Futhark's `[]`-polymorphic arrays without a shape DSL, it makes every operator shape-generic, and it removes static shape checking entirely. Rejected: it is a much larger change than the proposal, and it gives up the compile-time guarantee (§2 above) that motivates the work.

**So option A degenerates into option B, for a fixed, declared finite interface.** That is the central finding of this assessment, and it is why the recommendation below is a *table-level* change rather than an engine-level one. It is *not* a claim of equal expressive power in general: a finite class table cannot express open-ended dimensions, data-derived reshapes, or relations known only at run time. §12.5 and §12.6 state what is given up.

---

## 6. Option B, and the counts

### 6.1 The mechanism already exists

`Ty::Futhark(u16)` is a **table-defined type family**: one variant for the whole family, payload = the row index of `kingdoms/futhark_true/types.tsv` (`src/geneframe.rs:101-115`), with `MAX_FUTHARK_DUALS = 65 536` codes and a load-time guard (`check_dual_count`). A shape class is therefore *already* expressible as a type with no new machinery:

```
T0005   mat.n.m      mat   n,m   …   a 2-D array whose dims are the interface's n and m
T0006   mat.m.p      mat   m,p   …   …and the interface's m and p
T0007   mat.n.p      mat   n,p   …
```

Note the wording: **a shape class is a role in the interface, not a size.** `mat.n.m` does not mean "a 16×16 array"; it means "the array the interface calls `[n][m]`". The runtime binds the actual sizes; the Futhark source is emitted shape-polymorphically (`def main [n][m][p] (A: [n][m]f32) …`). This is the same move the kingdom already made once, and the precedent is worth stating precisely: `arr.f32` and `scn.f32` are **two duals for one Futhark shape**, split so that a property the language cannot see becomes a type. Futhark's `scan` returns an `acc f32`, and `map` over an `acc` yields an ordinary `[]f32` — measured (`/tmp/shape/acc_map.fut` compiles and returns `[2f32, 4f32]`) — so the kingdom's `scn.f32` tracks *"this value came from a scan"* through operations Futhark itself would widen. The precedent is therefore **"a property the language cannot see becomes a type"**, not "Futhark has a nominal type here".

### 6.2 The counts (computed with the engine's own bound formulas)

The bound formulas from `kingdom_contract.rs` §2.1–2.2 were transcribed into `/tmp/shape/poc_counts.py` and validated three ways before use: against the engine's published two-type fixture (`L=3`, `P=(3,2)`, `t=(1,1)`, `d=(3,2)`, width 5), against the single-type Karva-tail identity `t = d = h·(n_max−1)+1` for h ∈ {1, 7, 32}, and against **the engine itself** (§6.2.1).

**The shipped Futhark kingdom** (4 types, 38 rows), program size N = 3:

| root | frames | gene width | L | head alphabet sizes |
|---|---|---|---|---|
| `f32.real` | 4 | **40 words** | 9 | real 5, arr 8, scn 8, fn 17 |
| `arr.f32` | 4 | **42 words** | 10 | arr 8, scn 8, real 5, fn 17 |
| `scn.f32` | 4 | **42 words** | 10 | scn 8, arr 8, real 5, fn 17 |

#### 6.2.1 The same numbers, from the engine

```
$ cargo run --release --features wgsl --example futhark_true_emit -- --root f32.real --size 3 --count 1 --show 0
kingdom Futhark: 4 types, 38 rows, root f32.real, program size 3, gene width 40
$ ... --root arr.f32  →  gene width 42
$ ... --root scn.f32  →  gene width 42
```

The transcript and the hand-derivation are in `docs/reviews/2026-10-26_first_class_shapes_codex.md` §2 (that review computed 44/45/45 by taking `t_T = P_T - h_T` instead of `min(P_T, L + h_T) - h_T`; the engine settles it).

**The PoC, class-specialised.** Interface `A : [n][m]f32`, `B : [m][p]f32`; dimension variables `{n, m, p}`; operators `map`, `zipWith`, `reduce`, `transpose`, `matmul`; one terminal per class; a scalar class and a function class.

Three counts must be kept apart, and the review was right to insist on it:

| class set | kingdom types | **declared** shape classes | **root-reachable** frames | gene width | L | per-cell alphabet |
|---|---|---|---|---|---|---|
| **full** (all ordered pairs of `{n,m,p}`) | 14 | 12 array classes + scalar + fn | **10** (9 matrix + fn) | **285 words** | 38 | 7 per matrix class |
| **interface-minimal** (six matrices) | 11 | 6 + scalar + fn | **7** (6 matrix + fn) | **151 words** | 26 | 5 per matrix class |

Why the frame count is smaller than the class count: `KingdomContract::reachable` follows **argument** types only, so a class that is only ever an *output* — `scalar`, produced by `reduce` and consumed by nobody — is not reachable from a matrix root and gets **no frame**. The consequence is a real design fact worth stating: **`reduce` can only be the root of a scalar-rooted gene**; inside a matrix-rooted gene it would need a scalar slot, and there is none.

Row breakdown for the full set: `matmul` 27, `map` 12, `zipWith` 12, `reduce` 12, `transpose` 9, input terminals 14, scalar/const 2.

For scale: `kingdoms/wgsl_true` has **305 types and 8 841 rows**. The PoC is 11–14 types and 51–87 rows — one to two orders of magnitude smaller.

**Diversity does not collapse.** A matrix cell's alphabet is 5–7 symbols in the PoC against 8 for `arr.f32` today — the search is not reduced to one operator per class. (`matmul` contributes 3 of the 7 in the full set; `transpose`, `map`, `zipWith`, `reduce` and the input terminal contribute the rest.)

### 6.3 What the class set is, and what it costs

The class set is **not** the closure of the dimension space; it is the set of shape roles the program's interface mentions, chosen by whoever declares the interface. That is the answer to *"can we avoid generating concrete types for every possible dimension combination?"* — **yes, because we generate one type per role, and the roles are declared, not derived.** But the declaration is not free, and three properties must be owned by the interface designer:

- **Closure under composition.** A class set that is not closed under the operators' effects silently excludes legal programs. From `{mat(n,m), mat(m,p)}`, one `matmul` yields `mat(n,p)`, a `transpose` yields `mat(p,n)`, and the next legal `matmul` needs `mat(p,m)` or `mat(m,n)`. Pruning roles removes valid compositions; closing the set under `transpose` and `matmul` restores the larger table — over `{n,m,p}` that closure is all **9** matrix classes, not the 6 of the "interface-minimal" row. **The honest PoC class set is therefore the closed one**, and a pruned set is a deliberate restriction that must be recorded as such.
- **A terminal per demanded class — not a terminal per *input*.** `KingdomRelations::check` requires *"a terminal for every root and every required type"* (`kingdom_contract.rs:200`), and "demanded" is **global**: the roots plus every type any row asks for, not just the types reachable from one root. So every class named anywhere in the table needs *a* terminal. But a terminal need not be an external input: the kingdom already ships a **derived** terminal (`scn.f32.identity` = `map (\x -> x) (scan (+) 0.0 xs)`, a zero-argument terminal that produces a scan result). **How many external inputs a class set implies is a generator/interface choice, not a law** — and it is the number to watch, because `zipWith A B` needs *two* distinct arrays of one class, so terminals must be generated per *(input binding, class)* if the program is to see two different arrays of the same shape. That multiplies the class's alphabet and changes the count table.
- **A class the interface does not name cannot be used**, and the generator should *report* the classes an operator needs that the interface does not provide, rather than inventing them.

### 6.4 The growth, measured

The adversarial review was right that a single total width hides the shape of the growth. Measured over the full class set (all ordered pairs), N = 3:

| dimension roles | matrix classes | kingdom types | **rows** | root-reachable frames | **gene width** | L | max `t_T` | max `P_T` |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 8 | 39 | 5 | **95** | 18 | 18 | 23 |
| 3 | 9 | 14 | 87 | 10 | **285** | 38 | 35 | 37 |
| 4 | 16 | 22 | 163 | 17 | **635** | 66 | 63 | 65 |
| 5 | 25 | 32 | 273 | 26 | **1 193** | 102 | 99 | 101 |
| 8 | 64 | 74 | 867 | 65 | **4 595** | 258 | 255 | 257 |
| 10 | 100 | 112 | 1 543 | 101 | **8 783** | 402 | 399 | 401 |

Reading the table:

- **Rows are cubic in the role count** (`matmul` alone contributes `r³` rows; `map`/`zipWith`/`reduce`/`transpose` contribute `r²` each).
- **Width is quadratic in the role count** — the frames are `r² + 1`, and each frame's head grows with the program size, so the width is `Θ(r²)` with a slowly growing factor.
- **`t_T` grows with the role count, not just the frame count** — a new class raises `L` (the leaf bound), which raises `t_T` for *existing* frames. A new role therefore does not merely append a frame; it lengthens the tails of the frames already there. This is the review's "superlinear bounds" point, confirmed.

**The practical consequence: the design has a ceiling of roughly 3–5 dimension roles.** At three roles the PoC is 3.6× the shipped kingdom's width; at five it is 28×; at ten it is 208×. A class-specialised kingdom is a tool for **small, fixed interfaces** — a matmul-shaped problem, a stencil-shaped problem — not for a general array language. §12.6 says what to do when the workload is bigger than that.

---

## 7. What is affected, and what is not

**Not affected — the core GEP engine needs no change.** (This is the load-bearing claim, so it is stated per mechanism, with its precondition.)

| mechanism | file | why it is unaffected |
|---|---|---|
| `Frame`, `Span`, `DomainKey` | `typed_gep.rs:58-93` | keyed by `Ty`; a class *is* a `Ty::Futhark(row)`. |
| `alphabet` | `:223` | "symbols with output `T`" — with classes baked into the output, this is already the class-restricted alphabet. |
| `mutate` / `invert` / `swap_range` | `:239-263` | membership and domain checks; unchanged. |
| `validate` | `:524` | per-cell type equality; unchanged. |
| `express` (FIFO walk) | `:555` | the slot type comes from the signature; unchanged. |
| `place` / `fits` / `write_back` | `:629-697` | the round-trip checks; unchanged. |
| `required_heads`, `bounds`, `heads_from_program_size` | `kingdom_contract.rs:326-388` | keyed by `Ty`; a class set simply makes more frames. |
| Karva representation, epoch alphabet, ADBs, homeotic genes | `geneframe.rs`, `population_dag.rs` | orthogonal. |
| homologous crossover | `typed_gep.rs:176` | homologous = same root **class** + same `h_T` per class. **The proposal's §8 question "will homologous crossover continue to work when compatible basic types carry different symbolic dimension bindings?" — there are no symbolic bindings in the gene at all**; the binding was resolved into the type before the gene existed, so crossover exchanges cells of the same class and stays legal. |

**The precondition on that claim.** `Ty::Futhark(u16)` carries only a *table index*, not a kingdom or interface identity, and `homologous()` compares only the root and the head lengths. So two different interfaces that both number their first class `0` are, to the engine, the same type — and a crossover or a cached tree crossing the two would be semantically invalid. **A layout must therefore be immutably bound to one versioned kingdom/interface, and that identity must be part of the layout's identity** (so checkpoints, population sharing and crossover cannot cross the boundary). This is the one addition to the engine the design needs, and it is an identity check, not a mechanism change.

**Affected — five things, all outside the core engine:**

1. **The type table and the generator.** `types.tsv` gains one row per class; `functions.tsv` gains one row per *(operator, class)* instantiation; `gen_tables.py` gains the unification step that emits them. The proposal's §5 hybrid — *"parametric shape relationships might be resolved at the kingdom interface or gene-layout construction stage, producing a finite, shape-safe typed alphabet for each evolutionary run"* — is exactly this, and it is the recommended design.
2. **The Futhark rendering and the harness.** A class-typed tree must be rendered into shape-polymorphic Futhark: the source becomes `def main [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 = …`, and the `entry` point supplies concrete sizes. `futhark_kingdom.rs`'s `render` emits the *expression*; the wrapper that emits the `def`/`entry` is the harness's job and must learn the interface.
3. **The e-graph's datatype, ruleset *and* bridge.** §8. The bridge matters: `constructor_of`, `render` and `to_tree` are hard-coded per row name today (`futhark_egraph.rs:298`, `:328`, `:439`), so they must be generated too.
4. **The interface contract as a first-class artefact.** The class set must be recorded, versioned and shipped with the layout, because it *is* the program signature.
5. **The interpreter.** `futhark_interp.rs` must model the same shapes, or the differential test that currently proves the kingdom agrees with Futhark will be measuring two different languages.

**Not affected but worth naming:** the Futhark `acc` distinction survives untouched. `scn.f32` is a class like any other; adding `mat.n.m` does not disturb it. The e-graph's "the sort *is* the type" principle (§8) is what makes this work.

---

## 8. The e-graph

The e-graph's design principle — **one egglog sort per kingdom type**, so the sort *is* the type and egglog itself refuses an ill-typed program (`src/futhark_egraph.rs:8-13`) — extends to shape classes without any change of principle: **one sort per class**. `FR`/`FA`/`FS`/`FF` become `FR`/`FA_C…`/`FM_n_m`/…, and each constructor is per *(row, class)*.

The consequences:

- **The datatype grows with the number of generated rows, and the rows are not linear in the role count.** For the PoC: 87 rows → 87 constructors over 14 sorts. Egglog handles that comfortably; the current 38-constructor datatype is not near any limit. But `matmul` produces one row per consistent triple, so the row count is **cubic in the role count** (§6.4), and the datatype with it.
- **The ruleset is harder than "one copy per class".** The easy rules are the shape-preserving unary ones (`map id` → one copy per class). A rule that relates *two* classes — `transpose ∘ transpose`, `matmul ∘ transpose` — has to coordinate three or four linked role positions at once, and its instantiation count grows with the role count as well. The ruleset must therefore be **generated as relational schemas with exhaustive instance tests**, not hand-copied per class, and the generated rules need a saturation budget: the current `FUTHARK_RULESET` is already known to diverge when `map id` and the fusion rules are merged, and a class-per-sort ruleset multiplies the ways to get that wrong.
- **The guarantee is preserved and strengthened.** A rewrite cannot change a node's class, because the sort *is* the class; egglog refuses a term whose constructor's sort does not match its arguments. The proposal's invariant *"Fuller's e-graph rewrites preserve type and shape relationships"* therefore holds **by construction**, exactly as it holds today for the `acc` distinction.
- **The core write-back is unaffected; the bridge is not.** `place`/`write_back` use only row ids and `Ty`, and accept a class-specialised tree unchanged. But `import` maps a row *name* to a constructor through a hard-coded table (`constructor_of`), `render` is a `match` over constructor names, and `to_tree` maps back by name. Those three must be generated from the same row source, and the round trip `tree → e-graph → tree → write_back` must be tested for the generated rows.
- **The sort is not a shape checker.** Egglog preserves the relations the *table* states; it cannot tell that a mis-generated `matmul` row has its operands swapped. "The sort is the type" catches the *use* of a declared row, never a bad *declaration*. §12.2.

**Honest caveat:** a shape class is a *role*, and the e-graph does not know that `mat.n.m`'s second dimension is the same variable as `mat.m.p`'s first. That relation is enforced by the *table* (the `matmul` row's argument classes are `mat.n.m` and `mat.m.p`), not by the e-graph. That is where it belongs — but it means the e-graph cannot *invent* a cross-class relation, only preserve the ones the table states.

---

## 9. Option C, and why it is part of the answer rather than an alternative

Option C ("keep the existing type structure, provide high-level operators whose shapes are correct by construction") is the right *inventory* of operators and the wrong *mechanism*: without classes, `matmul` cannot be a row at all, because its two arguments have different shapes and the type system has one shape per type.

But C's insight — *the shape-sensitive operators are few, and they are exactly the ones that need classes* — is what keeps the class set small. In the shipped kingdom no row creates a competing array length: `map`, `map2`, `reverse` and `scan` preserve it, and `reduce` is rank-changing (array → scalar). Under a class scheme, the shape-*carrying* operators (`map`, `zipWith`, `reduce`) must become class-typed anyway, because the shape has to survive them for the sensitive ones to be usable.

The measured row counts (§6.2) show the split clearly: the shape-*sensitive* operators (`matmul` 27, `transpose` 9) are a minority; the shape-*carrying* ones (`map` 12, `zipWith` 12, `reduce` 12) are the bulk. That is the honest cost of making shape first-class: **shape-preserving operators become shape-typed**, because the shape has to survive them for the sensitive ones to be usable.

---

## 10. The minimum proof of concept

The proposal's §7 PoC is specified in *parametric* terms (`Vector(f32,n)`, `Matrix(f32,n,m)`). That tests option A, which §5 argues cannot work. **The PoC should test the recommended design instead**, which is a different experiment with the same seven measurements.

### 10.1 The kingdom

- **Interface:** `A : [n][m]f32`, `B : [m][p]f32`; dimension variables `{n, m, p}`.
- **Class set (declared, and closed under `transpose` and `matmul`):** the 9 matrix classes `mat(d1,d2)` for `d1,d2 ∈ {n,m,p}`, plus `scalar` and `fn` — **11 types, 75 rows, 10 root-reachable frames** from a matrix root, gene width 285 words at N = 3. *A pruned set (say 6 classes) is legitimate only if the restriction is recorded and the excluded compositions are known not to be wanted.*
- **Operators:** `map` (fn, C) → C; `zipWith` (C, C) → C; `reduce` (C) → scalar; `transpose` mat(d1,d2) → mat(d2,d1) for each class whose transpose is in the set; `matmul` mat(d1,d2) × mat(d2,d3) → mat(d1,d3) for each consistent triple in the set.
- **Terminals:** per *(input binding, class)* — at least two bindings for `mat(n,m)` and `mat(m,p)` so that `zipWith` can see two different arrays of one shape.
- **Rendering:** shape-polymorphic `def` with `[n][m][p]` parameters and an `entry` that supplies concrete sizes.

**A dead class in the proposal's own PoC.** The proposal's §7 kingdom lists `Vector(f32,n)` alongside `Matrix(f32,n,m)`, with the operators `map`, `zipWith`, `transpose`, `matmul`. In that operator set **the vector class is unreachable**: nothing in the set *produces* a vector, and nothing reachable from a matrix root *consumes* one, so `vec(n)` is never in `reachable(root)`, gets no frame, and cannot appear in any gene. (Measured: adding `vec(n)` to the class set leaves the reachable set unchanged.) The PoC must therefore either **drop the vector class**, or add an operator that bridges — a `dot : vec(n) × vec(n) → scalar`, or a vector-rooted gene. Shipping a class that no gene can mention is the shape-system equivalent of a dead row.

### 10.2 The seven experiments, re-specified

| # | experiment | pass criterion |
|---|---|---|
| 1 | Random gene construction | every gene expresses; **zero** `LayoutDefect`s over ≥ 10⁵ genes; no shape error is even *representable* |
| 2 | Mutation and crossover | every operator's draw is in the pre-enumerated alphabet; the expressed tree of every child is legal; crossover between homologous genes never produces an illegal gene; **no crossover crosses a kingdom/interface boundary** |
| 3 | Expression to a typed AST | the AST carries a class at every node; the class of a child is the class its parent's row declares |
| 4 | AST re-encoding into Karva | `fits` holds for every extracted tree; `write_back` round-trips (GEN → AST → GENE → AST is the identity) |
| 5 | E-graph import, rewrite, extraction | every rewrite preserves the class (the sort is the class); every extracted candidate is class-correct; the heritable set is non-empty for a non-trivial input; **the generated bridge round-trips** `tree → e-graph → tree → write_back` |
| 6 | Compilation through Futhark | **every** program compiles; a compile rejection naming a size is a **failure of the design**, per the proposal's rule, not a fitness outcome |
| 7 | Measurements | genome width, per-cell alphabet size, rows, frames, types, construction overhead — against the shipped kingdom's 4 types / 38 rows / 42 words and the PoC estimate of 11 types / 75 rows / 285 words |

**The two experiments that matter most** are 6 and 1. Experiment 6 is the proposal's own rule, and it is the one that distinguishes "shape is in the type" from "shape is in a comment": with classes in the type, Futhark's compile-time size check (§2 above) becomes *unreachable* for internal relations, because the table already guarantees them. Experiment 1 is the one that distinguishes legality-by-construction from legality-by-checking: the defect count must be zero, not small.

**One experiment the proposal does not list and should:** *the incompatible-shape test*. Attempt to construct a program where `matmul` receives `mat(n,m)` and `mat(n,p)` — the inner dimensions do not match. In the class-specialised kingdom this must be **unrepresentable**: there is no row, so no gene can hold it, and the mutation alphabet of the relevant cell simply does not contain `matmul`. That is the positive demonstration of the central claim, and it is the one the compiler cannot provide.

**Two more the review adds:** *the generator's own witness test* (for every generated row, assert the shape relation the row declares — that `matmul`'s first argument's trailing dimension equals its second argument's leading dimension — independently of the row text, so a mis-generated row cannot pass silently), and *the closure test* (assert that the class set is closed under every operator's effect, or record that it is not).

### 10.3 What the PoC must NOT do

- It must not require a change to `typed_gep.rs`, `kingdom_contract.rs` or `geneframe.rs`. If it does, the design has left option B and entered option A, and the analysis in §5 applies.
- It must not introduce a repair, a rejection filter or a substitution step. Any of those is a design failure by the proposal's own §1.
- It must not fix a concrete dimension. The classes are roles; the PoC's Futhark source is shape-polymorphic.
- It must not accept a class set that is neither closed nor recorded as deliberately pruned.

---

## 11. Invariant-by-invariant

The proposal's §6 list, checked against the recommended design.

| invariant | verdict | note |
|---|---|---|
| Every gene expresses a legal, shape-correct AST without repair or rejection | **holds** | the class is a type; the table is shape-correct by construction; the engine's (U1)+(U2) argument is unchanged. **The guarantee now rests on the generator, not only on the engine** (§12.2) |
| Mutation and crossover preserve legality by construction | **holds** | alphabets are per class and pre-enumerated; crossover exchanges same-class cells; crossover must not cross a kingdom/interface boundary |
| Karva remains the canonical genetic representation | **holds** | the gene is still symbol ids + constant indices; no new word |
| GEN → AST → GENE round-trip remains valid | **holds** | `place`/`write_back` unchanged; the round-trip is the existing one with finer types |
| E-graph rewrites preserve type and shape relationships | **holds, strengthened** | the sort is the class; egglog refuses a class-changing rewrite — but it preserves a *declared* relation, it does not validate it (§12.2) |
| COMPRESS / `write_back` preserve those relationships | **holds** | write-back is a per-type placement; classes are types |
| The number of reading frames and symbol rows remains manageable | **holds for small interfaces only** | 11 types / 87 rows / 10 frames for the PoC against wgsl_true's 305 / 8 841 / — but **width is quadratic in the role count** and rows are cubic (§6.4): the practical ceiling is 3–5 roles |
| Runtime dimensions may stay symbolic, but their relations must be guaranteed or declared | **holds, with a scope cut** | guaranteed internally by the table; declared externally by the interface. **Runtime-equality programs are out of scope** (§12.5) |

---

## 12. Risks, and what would falsify this

Stated plainly, so the design can be attacked before it is built.

1. **The generator becomes the new trust boundary.** The engine's guarantee used to rest on the engine's own type discipline. It now rests on the **table**: if the generator says `mat.n.m` and `mat.m.p` share `m` but renders an interface where those positions bind independently, or maps both to the wrong input names, then every gene is legal *relative to the table* and yet the program is wrong — or Futhark rejects it. The remedy is to make the table, the Futhark signature, the terminal binding and the interpreter's shape semantics **one checked artefact**, with a witness test per row and an independent shape-relation model in the generator. This is the review's sharpest objection and it is correct.
2. **"The sort is the type" is not a shape checker.** Egglog preserves what the table declares; it cannot detect a mis-declared row. The same remedy as (1): independent checking of the table, not of the e-graph.
3. **Terminals per input binding.** One terminal per class cannot express `zipWith A B` with two distinct arrays of the same shape. Generating terminals per *(binding, class)* restores expression and enlarges every class's alphabet — which changes the count table and the search space. This must be measured, not assumed.
4. **The class set does not close for free.** Pruning roles removes legal compositions; closing under `transpose` and `matmul` restores the larger table. The PoC must either use the closed set or record the pruning as a deliberate restriction.
5. **Runtime-equality programs are lost.** Two external vectors declared `[n]` and `[m]`, supplied with equal sizes at run time: the current `arr.f32` kingdom expresses `map2` over them and Futhark checks the equality at run time; two distinct static classes reject the program outright unless the equality is declared in advance. **This is a real reduction in expressible programs and it is a decision, not an oversight** — it must be taken consciously.
6. **The workload scope.** Batching, slicing, data-dependent dimensions, `reshape`, or a variable number of inputs either explode the table or silently exclude the productive programs. The architecture's honest answer is a **split**: class-specialised kingdoms for fixed-signature kernels, and a separate shape-aware IR/evaluator for dynamic-shape tasks. §13.
7. **Layout↔kingdom identity.** `Ty::Futhark(u16)` carries a table index, not an interface identity, and `homologous()` compares only the root and head lengths. Without a versioned kingdom/interface identity in the layout, checkpoints, population sharing and crossover can cross interfaces and produce semantically invalid programs.
8. **The growth ceiling.** Width is quadratic in the role count and rows are cubic (§6.4). The design is a tool for small, fixed interfaces.
9. **What would falsify the recommendation:** if the target programs need shapes that are *not* roles of a fixed interface — genuinely dynamic reshaping, a dimension computed from the data — then the class-set approach cannot express them, and the only remaining route is the runtime shape-carrying design (A2), which gives up compile-time shape checking. **That is the boundary of this design, and it should be stated before the PoC, not discovered after it.**

---

## 13. Recommendation

**Do not modify nucleotable's core. Do not build parametric reading frames. Build a shape-specialised kingdom — for small, fixed interfaces.**

1. Extend `kingdoms/futhark_true/gen_tables.py` to take a **declared interface** (dimension variables + shape roles, closed under the operators' effects, or deliberately pruned and recorded as such) and emit, for each class, the rows of every operator whose instantiation is consistent with that class set — one row per consistent instantiation, with the shared-dimension relations already resolved into the row's argument classes. Emit `types.tsv` with one dual per class and `functions.tsv` with the class-specialised rows. **The unification happens here, once, offline.**
2. Give the generator an **independent shape-relation model** and a **witness test per row**, so a mis-declared row fails the generator rather than the compiler (§12.1).
3. Keep the class set small and declared. Start from the closed set over 3 roles and report what the closure costs; do not pretend a pruned set is free.
4. Extend the renderer, the harness and the interpreter to the shape-polymorphic world: `def main [n][m][p] …` plus an `entry` with concrete sizes, with the interpreter modelling the same shapes so the differential test still means what it means today.
5. Move the e-graph's **datatype, ruleset and bridge** into the same generated output: one sort and one constructor per class, rules as relational schemas with exhaustive instance tests, and the constructor/renderer mappings generated from the same row source.
6. Bind a layout to one versioned kingdom/interface, and make that identity part of the layout's identity (§12.7).
7. Run the seven experiments of §10.2 — with experiment 1 (zero defects over ≥ 10⁵ genes), the *unrepresentable* incompatible-shape test, and the generator's witness tests as the three that decide the design.

**The honest summary of the trade:** this design gives up the *open-ended* parametric shapes of the proposal and the *generality* of option A, and buys back the thing the whole architecture is built on — **legality by construction, with no repair and no rejection, and no change to the core GEP engine**. For a kingdom whose purpose is to evolve programs that Futhark must accept, over a small fixed interface, that is the right trade. If the shapes needed turn out to be genuinely dynamic rather than interface roles, this design is the wrong one, and the assessment should say so before the PoC rather than after.

---

## Appendix A — the measurements behind this document

| measurement | where | how |
|---|---|---|
| The engine's own gene width for the shipped kingdom (40 / 42 / 42 at N = 3) | §6.2.1 | `cargo run --release --features wgsl --example futhark_true_emit -- --root <r> --size 3 --count 1 --show 0` |
| The bound formulas, transcribed and validated three ways | §6.2 | `/tmp/shape/poc_counts.py` — against the two-type fixture, the single-type Karva-tail identity, and the engine's own output |
| The PoC's counts (types, classes, frames, rows, width, L, alphabets) | §6.2 | the same script |
| The growth in dimension roles (2, 3, 4, 5, 8, 10) and the `P_T`/`t_T` maxima | §6.4 | the same script, `--growth` |
| `scalar` is reachable only as an output, so `reduce` can only root a scalar gene | §6.2 | `KingdomContract::reachable` follows argument types only (`kingdom_contract.rs:298`) |
| Futhark rejects a statically-known size mismatch at compile time | §2 | `/tmp/shape/futhark_shape_transcript.txt`, Futhark 0.28.1 |
| Futhark accepts dynamic sizes and checks them at runtime | §2 | the same transcript |
| `map` over a scan result yields an ordinary array in Futhark | §6.1 | `/tmp/shape/acc_map.fut` in the same transcript |
| The kingdom's failures are hangs, not compile errors | §2 | `docs/RESCUE_futhark.md` §4 (8/8 candidates timed out at ~120 s), §6.1 (three compilers) |
| The adversarial review and the disposition of every finding | throughout | `docs/reviews/2026-10-26_first_class_shapes_codex.md` |

## Appendix B — file and line references

- `src/typed_gep.rs:58` `Frame` · `:97` `GeneLayout` · `:176` `homologous` · `:223` `alphabet` · `:239` `mutate` · `:249` `swap_range` · `:524` `validate` · `:555` `express` · `:615` `required_heads` · `:629` `place` · `:678` `fits` · `:688` `write_back`
- `src/kingdom_contract.rs:34` `KingdomRelations` · `:84` `check` · `:58` `a_{T,U}` · `:63` `a_T` · `:65` `n_U` · `:162` `a_tu`/`a`/`n_u` derivation · `:190` the *global* demanded set · `:200` the terminal-for-every-demanded-type rule · `:298` `reachable` · `:326` `heads_from_program_size` · `:371` `bounds`
- `src/geneframe.rs:46` `Ty` · `:115` `Ty::Futhark(u16)` · `:157` `Ty::code` · `:190` `MAX_WGSL_DUALS` · `:197` `FUTHARK_CODE_BASE` · `:204` `check_dual_count`
- `src/futhark_kingdom.rs:13` the "shape is a value, not a type" decision · `:45` `FutharkDual` · `:188` `relations` · `:205` `render` · `:308` the four-type/38-row assertion
- `src/futhark_egraph.rs:8` one sort per type · `:88` `FUTHARK_DATATYPE` · `:148` `FUTHARK_RULESET` · `:253` `constructor_of` · `:298` `import` · `:328` `render` · `:439` `to_tree`
- `nucleotable/docs/MultiTypedGEP.md:15` the contract (§1) · `:32` the layout and bounds (§2) · `:69` the FIFO expression (§3) · `:193` variation enumerates its domain before the draw (§5)
