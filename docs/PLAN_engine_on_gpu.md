# Plan: the evolution engine in Rust and WGSL

## Context

### The objective

**Beat the SRBench benchmarks with HFF-SR.** At noise 0 the only published
method above us is AIFeynman (54.1%). Our development races (our own seeds,
never SRBench's) stand at 46–50 of 133 (35–38%) with 30 seconds to a few
minutes per problem, where the published methods had up to 8 hours.

### The approach, and the unique feature

GEP chromosomes (K-expressions, multi-gene, islands, the pump) selected by HFF,
the hyperspherical fitness function, over train, validation and edge-validation
errors. fuller replaces sympy, and — the unique feature — **fuller's simplified
forms go back into the genes**, so simplification is a variation operator, not a
report-time clean-up.

### Where the time goes (MEASURED, 2026-09-20/21)

One generation of 800 individuals takes about 1 second. The GPU join (scoring)
is 2% of it. The other 98% is Python: geppy objects cloned and mutated one at a
time (35% + operators), snap and physics per offspring (38%), host-side token
conversion and fitness assignment (16%).

A numpy prototype of the variation phase (`hff/notebooks/_tensor_pop.py`, the
population as integer tensors) ran selection + cloning + all 12 GEP operators in
**12.6 ms** per generation for 600 individuals, and took a 30-second fit from 10
generations to 28. It proves the layout and the operator rules. **It is a
throwaway: this is a Rust and wgpu project, and numpy, geppy, deap and sympy are
legacy to be taken off the stack.** The prototype and its tests are the
executable specification the Rust is checked against; then the Python goes.

### Why this plan

Every idea we want to test — more generations, bigger populations, harvest and
regrow, ten seeds per problem — is priced in generations. At ~5 ms a generation
(the floor is one join dispatch, ~4 ms for 800 individuals × 5,750 rows) a
10,000-generation run is under a minute instead of 3–4 hours. GEP is unusually
well suited: a chromosome is a fixed-length row of integer tokens, so every
operator is indexing, never tree surgery.

### What success looks like

1. The whole generation loop — select, vary, score, HFF — runs on the device
   with no Python in it, bit-reproducible from a seed.
2. A race on a development seed solves at least what the Python engine solves.
3. Python is reduced to the `fit`/`predict` wrapper SRBench requires.

## The layout (fixed now; every kernel reads it)

```
genome[P × G × (H+T+D)]  u32   token ids: head, tail, then the Dc domain's
                                indices — geppy's own gene layout
rnc[P × G × L]           f32   each gene's random numerical constants
fitness[P]               f32   NaN = not evaluated
wrapper_id[P], linker_id[P], a[P], b[P]
```

Token ids come from a per-fit symbol table keyed on fuller's `semantic_id`
(`geneframe.rs`), with `arity[]`, `is_function[]`, and the two sampling lists
(functions; terminals less the ones withheld from sampling — named constants
are graftable by snap, never drawn).

**Randomness** is counter-based: a draw is `hash(seed, generation, row, slot,
stream)`, the same arithmetic in Rust and in WGSL. No state, no thread-order
dependence; a CPU reference reproduces the device bit for bit, which is how
every kernel is tested.

## Steps

| # | What | Done when |
|---|---|---|
| 1 | **Resident population + RNG.** `src/evolve/`: layout, symbol codes, the hash RNG in Rust and WGSL, an `init` kernel, read-back. | device population == CPU reference, bit for bit; structural rules hold (tail = terminals only, Dc in range, withheld never drawn) |
| 2 | **Variation kernels.** Tournament (two per island + elites), uniform mutation, inversion, IS / RIS / gene transposition, the four Dc operators, one-point / two-point / gene crossover — geppy's rules exactly (`_tensor_pop.py` is the spec). | each kernel == CPU reference on the same draws; rules hold through 200 generations |
| 3 | **The loop on the device.** Genome buffer → the existing evaluator (`gpu_eval.rs`) with no host token lists → link/wrap/scale (`chrom_score.rs`, moved to WGSL) → HFF with the constant-model caps (from `hff_core`) → fitness buffer → step 2. One command buffer per generation; read back only the best rows. | N generations with zero host work; fitness == the Python engine's on the same population |
| 4 | **Pump, dedup, hall of fame.** Row hash for dedup; top-k; promote 2, keep 20%, refill by the `init` kernel; cross step; park / resume = buffer read / write. | same population dynamics as the Python pump on a fixed seed; resume is exact |
| 5 | **fuller's operators as kernels.** The linter kernel already runs on the device (113,444 expressions, 0 differences, 0.25 s); add snap (binary search in the sorted constant table + graft) and the physics rules as rule-table rows with a bounded-growth slot. Write-back into the gene on the device. | simplified genes re-enter the population with no host round trip |
| 6 | **Python off the stack.** The final model: fuller's final form in Rust (f64), a Rust infix printer (exists: `lint/node.rs`). The SRBench wrapper calls one Rust `fit`. Delete `_tensor_pop.py`, the geppy path, and the sympy extraction. | a race runs with geppy, deap, numpy-in-the-loop and sympy uninstalled from the hot path; solves hold |

## Rules of the build

- Rust and WGSL only. No new numpy. `RUSTFLAGS="-D warnings" cargo test` and
  `cargo clippy --all-targets` (plain and `--features gpu`) clean before every
  commit; never `#[allow]`.
- The device never decides anything in f32 that leaves the device: anything
  reported or written back as a constant is re-derived in f64 on the host
  (the rule the linter kernel already follows).
- Every kernel has a CPU reference in Rust and a bit-exact parity test.
- Development runs use our own seeds; SRBench's are for the final evaluation.
- Local commits, no push.
