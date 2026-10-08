# The WGSL kingdom

Status: specification, 2026-10-08. No rows loaded, no kernel. The type table
is in `TYPES.md`; the symbol rows will be `symbols.md` when written.

## Context

**Objective.** Evolve GPU compute code: WGSL kernels, as wgpu runs them on
Metal, Vulkan and DirectX 12. Correctness is a gate (the kernel must match a
reference output), efficiency is the fitness (time, memory traffic, energy
under HFF). The economic target is inference and our own engine's kernels; a
kernel found once runs billions of times.

**Approach.** The kingdom's genome is the pure expression DAG a kernel
computes between its loads and its stores. naga, wgpu's shader front end,
already holds a kernel that way: an arena of expressions referenced by
handle, with control flow as a separate statement tree. A head gene is one
root, a stored value or a branch condition; the loops and branches around it
are a fixed scaffold that acts as the chromosome's linker; shared expressions
become homeotic tail genes (`docs/Geneframe_Homeotic_Genes_design.md` §7).
Existing kernels are read into this form first, round-tripped through naga
and a parity run, so the representation is learned and checked before
anything is evolved.

**Unique feature.** Two-column typing. A row names its *slot* type (what the
hardware stores: `f32`, `u32`, `vec4<f32>`…) and its *content* type (what the
value means: `real`, `fixed Q15.16`, `index`, `code`, `bits`, `hash`…). The
same `u32` slot carries five contents with disjoint operator sets; a gene that
mixes them has no row, so Design C's projection substitutes the fallback and
the error is unrepresentable rather than a bug. Pack and unpack rows are the
only crossings between contents; `as` and `bitcast` the only crossings between
slots.

**Why now.** The share finder, the homeotic fold and the DAG extractor exist
and are measured; naga's IR is DAG-shaped by construction; and the measured
kernels (below) show the shape the kingdom has to express.

## Name

`WGSL`, after the language. wgpu is the runtime, naga the compiler; the genes
are WGSL.

## What a kernel looks like as a DAG (measured)

`examples/wgsl_dag_study.rs` parses a kernel with naga and hash-conses each
function's expression arena, operands by canonical id, effectful results
(loads, call results, atomics) kept unique.

| kernel entry | expressions | distinct | repeats | loads/effects | stores | if | loop |
|---|---|---|---|---|---|---|---|
| phylu `decode_main` | 864 | 489 | 375 | 247 | 125 | 43 | 9 |
| phylu `score_main` | 1,106 | 651 | 455 | 400 | 115 | 51 | 31 |
| phylu `mutate_main` | 836 | 480 | 356 | 269 | 76 | 48 | 26 |
| phylu `compile_main` (regex) | 832 | 407 | 425 | 207 | 87 | 66 | 16 |
| phylu `hff_main` | 303 | 180 | 123 | 69 | 38 | 18 | 3 |

Readings: naga's IR is already a DAG, so hash-cons-then-extract applies
directly; the 40–50 % repeats are mostly address arithmetic and literals the
device compilers already value-number, so they say what the kingdom optimises
over, not that these kernels are slow; and a quarter to a third of every
kernel is loads, with the stores, ifs and loops marking where the pure
expression DAG ends and the fixed scaffold begins.

## Reading a kernel into Karva

1. **Cut at the effects.** Every `Store` value and every `if`/`loop`
   condition is a root. Loads, literals and function arguments are the typed
   leaves. The statement tree is the scaffold, kept verbatim with one hole per
   root.
2. **One root, one head gene.** Each root's expression tree is encoded with
   `karva::terms_to_karva` over the WGSL symbol table. Expressions shared
   across roots become tail genes read through `href<t>`
   (`homeotic::fold`).
3. **Round trip is the test.** Chromosome → naga `Module` → WGSL text →
   naga validation → compile → a parity run against the original kernel on
   recorded inputs. Bit-exact on the same device.

## Evaluation path (the kingdom's kernel)

Unlike the other kingdoms, an individual is evaluated by compiling it and
running it: naga validation (free, filters most mutants), pipeline creation
(cached by source hash), a correctness run against the reference output on
the recorded inputs (the gate), then timing as the median of repeated runs
(the fitness, with its variance as a second objective). Slow per individual,
repaid because the product runs billions of times.

## What is built, what is not

| piece | status |
|---|---|
| naga reader and DAG measurement (`examples/wgsl_dag_study.rs`) | built |
| type table (`TYPES.md`) | specified |
| symbol rows with slot × content arity | not started |
| naga ↔ Math-shaped datatype so fuller's encoder and e-graph see a kernel | not started |
| the scaffold-with-holes chromosome and its decoder (phylu) | not started |
| compile-run-time evaluation path with the correctness gate (phylu) | not started |
| first target: one of our own kernels, read in, round-tripped, then evolved for time | not started |
