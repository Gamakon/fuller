# `futhark_true` — a Futhark-style parallel kingdom

**What it evolves:** a typed, high-level *parallel program* over arrays — a
tree of combinators (`map`, `map2`, `scan`, `rev`, `reduce`), each applied to
a function value or to another array. The program is compiled by Futhark, so
the GPU code that runs it is Futhark's, not ours.

**Why it exists.** `wgsl_true` is a legal and expressive GPU language, but its
search space is the wrong shape. Of its 305 types, 112 are `stmt.*`/`block.*`
scaffolding — rung × depth × fuel stratum — and **5,295 of its 8,841 rows
(59.9 %) produce one of those types**. Three fifths of the genetic material
of `wgsl_true` buys ways to *spell a statement nest*, not ways to compute
anything. That is why evolution finds legal syntax and not speed.

This kingdom is the same idea in the opposite shape: a tiny type system, a
tiny symbol table, and a compiler that does the optimisation evolution cannot
discover (fusion, vectorisation, tiling, size polymorphism).

| | `wgsl_true` | `futhark_true` |
|---|---:|---:|
| types | 305 | **4** |
| symbol rows | 8,841 | **38** |
| rows that are syntactic scaffolding | 5,295 (59.9 %) | **0** |

## The typing rules

Four types. Every rule below is carried by a type, so an illegal program is
*unrepresentable* rather than rejected — there is no repair, no refusal, no
substitution anywhere in this kingdom.

| dual | meaning |
|---|---|
| `f32.real` | a real number |
| `arr.f32` | a one-dimensional array of reals that is **not** a scan result |
| `scn.f32` | a one-dimensional array of reals that **is** a scan result |
| `fn.f32_f32` | a function from a real to a real |

**Shape is a value, not a type.** An array's length is not in the type system,
so `arr.f32` is one type however long the array is. This is the single
decision that stops the type count from following the program size — the thing
`wgsl_true` could not do, because a WGSL kernel's shape and its statement
nesting *are* in its type system.

**Nesting is free.** `map`, `map2` and `scan` are value-level combinators, so
a program's depth costs no types at all.

**A function value is a type, not a lambda.** `fn.f32_f32` is a type with its
own reading frame, holding the elementary functions. `map` takes one. So the
GEP never needs a bound variable: an elementary function's *parameter* is bound
by its spelling (`\x -> f32.sqrt x`), and its *argument* — a captured
literal — is an ordinary child slot (`\x -> x + {0}`).

**A reduction is a fused row.** `reduce` needs a two-argument combiner, and
the GEP's argument slots are values, not binders. So the combiner family is
spelled out: `reduce_add` and `reduce_mul`, over each of the two array
flavours. Fusing costs four rows and buys a first-order type system.

**Every neutral is the operator's identity.** `reduce (+) 0.0`,
`reduce (*) 1.0`, `scan (+) 0.0`, `scan (*) 1.0`. This is Futhark's own
precondition, not a taste. Futhark's manual: *in an expression `reduce f ne
xs`, the function `f` must be associative and have `ne` as neutral element …
if we … do not have the provided neutral element, we will get wrong results at
run-time … the compiler will not be able to detect that we messed up.* It is
not hypothetical. Measured on Futhark 0.28.1, over `[1,2,3,4]`:

| expression | right answer | Futhark 0.28.1 |
|---|---:|---:|
| `scan (+) 0.5 xs` | `[1.5, 3.5, 6.5, 10.5]` | `[1.5, 3.5, 6.5, 10.5]` ✔ |
| `reduce (+) 0.0 (scan (+) 0.5 xs)` | 22.0 | **27.0** ✘ |
| `reduce (+) 0.0 (scan (+) 0.0 xs)` | 20.0 | 20.0 ✔ |
| `reduce (+) 5.0 (scan (+) 0.0 xs)` | 25.0 | **45.0** ✘ |

The excess is the neutral counted once per element instead of once. Worse, the
error **poisons its siblings**: in a program holding both
`last (scan (+) 0.5 xs)` and a reduce over that scan, `last` returns 12.5 where
10.5 is right. With every neutral at its identity, all of sizes 1…9 are exact.

Two consequences for the table:

* the reduce and scan rows carry **no** neutral slot, so the gene cannot
  choose an undefined one; and
* there is **no `reduce f32.max`/`reduce f32.min`**. Their identity is
  ∓infinity, Futhark has no infinity literal (`f32.infinity` is an unknown
  name), so a max reduction has no neutral element that is both exact and
  defined. The elementwise max and min survive as `map2`, which takes no
  neutral at all.

**One extra type buys the whole discipline.** `scn.f32` is "a scan result". A
`scan` takes an `arr.f32` and gives an `scn.f32`, so **scan-over-scan is
unrepresentable**. That matters: Futhark 0.28.1 has an *internal compiler
error* on `scan (+) 0.0 (scan (+) 1.0 xs)`, and about 5 % of random genes
otherwise hit it.

**The two array flavours track Futhark's `acc` type exactly.** Futhark
distinguishes a scan result (an `acc` array) from an ordinary array, and the
distinction is not cosmetic: `reduce` over an `acc` array takes the other
code path, the one with the neutral bug above. Measured:

| expression | result is an `acc` array? |
|---|---|
| `scan f xs` | yes |
| `map f (scan f xs)` | yes |
| `map2 f (scan f xs) (scan f xs)` | yes |
| `map2 f (scan f xs) xs` | yes |
| `reverse (scan f xs)` | **no** |
| `map f xs` | no |

So `map` and `map2` keep the flavour and **`reverse` drops it**: `rev_s` is
typed `scn.f32 → arr.f32`, not `scn.f32 → scn.f32`. A scan result can still be
mapped, zipped and reduced; it just cannot be reversed back into a scan
result, because Futhark will not let it.

## The roots

| root | the program |
|---|---|
| `arr.f32` | an ordinary array-to-array program |
| `scn.f32` | an array program whose result is a scan |
| `f32.real` | a reduction, to a single real |

## What is here

```
README.md        this file
gen_tables.py    the generator: the design written as code
types.tsv        the types
functions.tsv    the symbol table: output type, ordered argument types, spelling
```

The table is the design; the Rust is its loader:

* `src/futhark_kingdom.rs` — the loader, the kingdom's relations, and the
  renderer that turns an expressed tree into Futhark source.
* `src/futhark_interp.rs` — the kingdom's own interpreter: what a gene *means*,
  independent of Futhark.
* `src/futhark_exec.rs` — compile the rendered source with the real Futhark
  compiler and run it.
* `examples/futhark_true_emit.rs` — the end-to-end check: random genes in,
  Futhark source out, the real Futhark compiler's verdict, and the
  interpreter's value compared with the compiled program's.

```
cargo run --release --features wgsl --example futhark_true_emit -- \
    --root arr.f32 --size 4 --count 100 --check
```

### What was measured

12 random genes for each of the 3 roots × 4 program sizes (2…5) — 144
programs, each compiled with the real Futhark 0.28.1, linked with `cc`, run,
and compared with the kingdom's own interpreter on a fixed 16-element input:

| | |
|---|---:|
| programs compiled, run and **in agreement with the interpreter** | **143 / 144** |
| legal programs the compiler does not converge on | 1 / 144 (0.7 %) |
| random genes refused, repaired or substituted | **0** |
| rows that are syntactic scaffolding | **0** |

### One primitive needed care

Futhark's `f32.round` rounds a half to the nearest **even** — it is `rintf`,
and WGSL's `round` is ties-to-even too. Rust's `f32::round` rounds a half
**away from zero**. On `round(2.5)` that is a whole unit, and it propagates:
`map2 (*) (map f32.round xs) xs` followed by `f32.tan` gives 6.765 where
Futhark gives −8.451. The interpreter uses `round_ties_even`.

### What the target language does not guarantee

The kingdom guarantees that every gene denotes a **legal, well-defined**
program. It cannot guarantee that Futhark's optimiser *terminates* on every
such program, because that is a property of Futhark. Measured on Futhark
0.28.1: a program that reduces over a `map2` whose second operand is a
reversed scan-derived array, and whose first operand contains a reduce over a
scan result, **never finishes compiling** — killed at 300 s with no output.
Removing either feature brings it back to 0.1 s.

So the compile step needs a wall-clock bound: `src/futhark_exec.rs` enforces
120 s, and a program that exceeds it is a *failed compile*, not a failed gene.
The reproducer is in `docs/ASSESSMENT_futhark_kingdom.md` §10.6.

## What is not here yet, and why

* **`filter`.** Futhark's `filter` returns an array of *unknown* size, and
  Futhark cannot zip an array of unknown size with one of known size. A
  kingdom that offered both `filter` and `map2` would therefore be able to
  express programs Futhark rejects — a hole in "legal by construction". The
  fix is a second shape dimension (`uns.f32` alongside `arr.f32`), with
  `map2` restricted to the sized flavour: one more type, and the hole is
  closed. It is not in v0 because it doubles the array rows and v0 is
  deliberately the smallest thing that is *entirely* legal.
* **`gather`.** Futhark's array-index gather (`xs[is]`) does not unify its
  size with the index array's in a way Futhark 0.28.1 accepts; the index
  array's size has to be a literal.
* **`scatter`.** Futhark's `scatter` is destructive; expressing it needs an
  array *and* a value, and Futhark's in-place update needs the array's
  contents to be unique. A later version, with the `uns.f32` dimension.
* **f16, i32, u32, 64-bit.** One element type in v0. Each costs one type and
  a factor on the rows.
* **The GPU path.** v0 renders to Futhark *source*, and Futhark's released
  compilers (C, Haskell, Python, LLVM) run it. Futhark's WGSL backend is
  experimental and PR-only (diku-dk/futhark#2140); see
  `docs/ASSESSMENT_futhark_kingdom.md` for what was measured about it,
  including the two places its WGSL does not survive naga and the one place
  it does not survive the WGSL specification.

## The genetic representation

The Karva gene is the program. The Futhark source is a *rendering* of the
expressed tree, regenerated on every evaluation; nothing reads it back into
the genome. The renderer is `FutharkKingdom::render`, and it is a pure
function of the tree and the symbol table.
