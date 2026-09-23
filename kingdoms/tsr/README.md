# TSR — Transcendental Symbolic Regression

**Andrew Morgan's name for it**, and the name carries the claim:

> "It's the only symbolic regression to properly handle the distance needed
> between the functions."

Everyone writes SR for symbolic regression. This is **TSR**.

## What is different

Ordinary SR gives every function the same type. A float goes in, a float comes
out, so `exp` will accept `tanh`'s output as readily as it accepts `x`, and the
search is free to build `tanh(exp(cos(log(x))))`. Nothing in the representation
says that is a worse shape than `m*v**2/2` — only the fitness does, after the
fact, by which point the head space is already spent.

TSR types the **distance** between functions. A transcendental's output is not
the same type as a plain float, so it cannot be fed back into another
transcendental without limit. The depth a term sits at becomes part of its
type, and a term too deep has no signature at all.

Illegal shapes stop being penalised and become **unrepresentable**.

## The types

| type | meaning |
|---|---|
| `F` | a plain float: a variable, a constant, or arithmetic over plain floats |
| `T1` | the output of one transcendental over plain floats |
| `T2` | the output of a transcendental over something already `T1` |

There is no `T3`. **That absence is the depth rule** — not a check, not a
penalty, an arity signature that does not exist.

- **Arithmetic ABSORBS**: `+ − × ÷` yield the highest depth among their inputs,
  so depth propagates rather than resetting. `tanh(x + exp(y))` is `T2`.
- **Transcendentals RAISE**, and have no row accepting `T2`.

Full design: `docs/SPEC_typed_transcendental_depth.md`.

## Why two, and not one or three

Measured across all 133 SRBench ground-truth models (128 parse):

| | count |
|---|---|
| **directly nested transcendental pairs** | **0** |
| laws at depth 0 | 78 |
| depth 1 | 47 |
| depth 2 | 3 |
| depth 3 or more | **0** |

A ceiling at `T1` would exclude three real laws. A ceiling at `T3` would admit
shapes no law uses. **`T2` is where the data puts it.**

The three depth-2 laws get there *through arithmetic*, never by direct nesting
— which is why arithmetic must absorb rather than reset.

## Validation, before any engine work

| check | result |
|---|---|
| all 128 parseable true laws type cleanly | **128 OK, 0 rejected** |
| our 8 reported models: do the bloated ones type? | **7 of 8 ILLEGAL** (213–412 chars) |
| the one legal exception | 60 chars, depth 1 — the run that scored test R² 1.000000 |
| share of a live population's sub-expressions that would be illegal | **23%** (400 real fold events) |

The 23% is biased toward the fold operator's targets; a uniform gene dump would
give a truer figure.

## Relationship to the `symbolic-regression` kingdom

`symbolic-regression/` is the untyped float op set the engine has always run —
the control. TSR is the same op set with the typing. They are two phylogenetic
spaces over one engine, which is the harness claim in `../README.md`: change
the symbol table, not the search.

That makes the A/B clean. Same engine, same selection, same pump, same islands,
same HFF; only the table differs.

## How it is built

`Config::typed_depth: Option<u32>`, **default `None`** — `None` is the
`symbolic-regression` kingdom, `Some(2)` is this one. `EVOLVE_TYPED_DEPTH`
names it and the run card carries it, so an A/B's two arms differ by one line
of the card diff. `Some(0)` is refused at `Engine::new` rather than read as
"off": a ceiling of zero is a search with no transcendental in it, which is a
different thing.

The symbol table is `geneframe::typed_depth_table()` — its own kingdom, so the
untyped rows are literally untouched. `symbols.md` states it row by row.

**The engine spends a depth budget top-down rather than matching output types
slot by slot**, which is the same predicate: both compute the largest
transcendental count on a root-to-leaf path, which is what `engine::t_depth`
already measured bottom-up. One `u32` per symbol (`depth_cost`) and one extra
draw list (`sample_flat`) is the whole cost, as the spec estimated.

Typed: `init` and point mutation, on the GPU and in the CPU reference, checked
against each other bit for bit. Not typed, on purpose: the span-moving
operators, which are left to the decode — see `symbols.md`.

Two things in the spec did not survive being built, and both are corrected in
`docs/SPEC_typed_transcendental_depth.md`: there is **no existing arity walk**
for the sampler to read a demanded type out of (`decode_gene` runs on the host,
long after the kernel has filled the row), and a typed refusal is **not** what
the engine reports as `oversized` (that is a gene that decoded and was too
big).

## Status

Built and tested; the A/B is running. The settings are the ones that recovered
75 of 133 — 800 intake + 400 champion, pump every 100, `cohort_merge` 10,000,
gene subsets on, no SMOGD/SMOTE — over three development seeds, with time
binding so a difference in ms/generation shows up as generations.

Results land in `measurements/` with their run cards.
