# The safeWGSL Kingdom: a Typed, Fully Evolvable Grammar of GPU Compute Kernels for Gene Expression Programming

Andrew James Morgan, Gamakon Ltd., London. First full version, 2026-10-13. House style follows `papers/geneframe-gecco.tex` (design is the contribution; results are highlights with their conditions).

Every claim about the system below cites a file and line in `/Users/andrewmorgan/Dev/gamakon/fuller` (or `../phylu`) or a commit hash in those repositories. Numbers come from commit messages, the documents named, or the run logs under `logs/HGP/001`, and say where.

---

## Context

**Objective.** Evolve GPU compute kernels, written in WGSL as wgpu runs them on Metal, Vulkan and DirectX 12, with gene expression programming: a fixed-length Karva genome that decodes to a kernel, raw genetic operators on that genome, and a fitness that compiles the kernel, checks it against a reference and times it. Correctness is a constraint; speed is the objective. The economic target is a kernel found once and run billions of times (`kingdoms/safewgsl/README.md`, Context).

**Approach.** The kingdom is a table of types and typed rows in the geneframe, the master symbol table shared by every kingdom of the engine (`docs/PLAN_population_dag.md`, Context; fuller `CLAUDE.md`, Architecture). Its types are *duals*, `slot.form`: what the hardware stores and what the value means (`kingdoms/safewgsl/TYPES.md`, opening). Its rows are the whole WGSL compute language, statements included, so a gene is a complete kernel, not an expression hung in a fixed scaffold (`docs/kingdom/WGSL_COVERAGE.md`, Context). The duals are chosen so that a kernel that crashes, hangs, races or writes another thread's cell has no spelling (`kingdoms/safewgsl/gen_tables.py`, header: "solve all these kernel crash issues by being inventive with types"); a layered checker proves, after every rewrite, what the types alone do not (`src/wgsl/check.rs:1-14`). Variation is typed: a knob mutation that changes a tuning parameter and every position that depends on it in step, structural rewrite rules with preconditions, generic pattern-to-action rewrites, typed subtree crossover, and the engine's own point mutation, all on the same genome, every child through the checker before it reaches the GPU (`src/wgsl/knob.rs:1-20`, `src/wgsl/rules.rs:1-12`, `src/wgsl/rewrites.rs:1-10`, `../phylu/src/evolve/wgsl_cross.rs:1-15`).

**Unique feature.** Two-column typing. A `u32` slot carries `index`, `count`, `bits`, `hash32` and six code forms with disjoint operator sets; a gene that mixes them has no row, so the total typed projection of the geneframe substitutes the form's fallback and the error is unrepresentable (`kingdoms/safewgsl/TYPES.md`, Forms table and the fallback paragraph). The same move types the kernel's structure: a statement's dual records its loop depth and whether it stands under a condition, a barrier row exists only at unconditional depths, a loop bound is a dual produced only by the bounded constructors, a store to the output accepts only the dual `cell`, "an output cell index owned by this thread" (`kingdoms/safewgsl/TYPES.md`, Forms: `cell`, `bound`, `C0`..`C4`, `L0`..`L4`).

**Status, 2026-10-13.** 183 duals and 1,529 function templates (`kingdoms/safewgsl/types.tsv`, `kingdoms/safewgsl/symbols.md`, last line). 181 of 184 WGSL construct test kernels read, render and validate; the three left are constructs naga 0.20 cannot parse (`docs/kingdom/WGSL_COVERAGE.md`, Status). 1,000 of 1,000 random trees of the kingdom's sampler render, pass the checker, compile on Metal and read back (commits 51eb0ee, 58dc7e1); 1,000 of 1,000 survive encoding to a Karva genome and decoding (commit ceb935f); 123 of 1,000 return to the identical tree after emit and parse (commit 3334408). Sixteen structural rules and twelve generic rewrites exist, every child through the checker (`src/wgsl/rules.rs:1811-1830`, `src/wgsl/rewrites.rs:18-31`). A rule-driven run from model-written seeds improved the best kernel from HFF 0.1192 to HFF 0.1052 in four generations, through a lineage of three rule steps, with 180 of 180 rule children feasible (`logs/HGP/001/fit_RA_rules/ledger.tsv`; `logs/HGP/001/fit_RA_rules.log`, closing lines); re-timed in six interleaved rounds, the generation-0 best ran at 8.9 / 8.4 / 8.3x the shipped kernel and the final best at 10.5 / 10.2 / 9.8x, faster in 6 of 6 rounds at every shape (`logs/HGP/001/fit_RA_rules/retime_paired/`). Two random-start runs of 65 and 59 generations found no feasible kernel (`logs/HGP/001/fit_SF_random_stopped_emptypump/generations.tsv`, `logs/HGP/001/fit_SF_pump_paused_g59/generations.tsv`).

**Why.** Evolution can only find what the kingdom can spell, and can only be trusted with what the kingdom cannot mis-spell. A kernel that compiles and runs but hangs the device, races on a cell or reads past its buffer is not an individual with low fitness; it stops the run and, on 2026-10-10, twice required a machine restart (commit d79d6f7). The kingdom's answer is to make those kernels unconstructible by type, prove the rest by a checker, and give every remaining individual a graded fitness, feasible or not, so selection can climb.

---

## 1. Introduction and contribution

Gene expression programming (Ferreira, 2001) encodes a program as a fixed-length string whose head may hold functions and terminals and whose tail holds terminals only; read breadth-first the string always decodes to a complete tree, so mutation and crossover act on the string and every child is a program. The geneframe lifts this to many value types: each symbol carries a typed arity signature, a kingdom is a query over one master symbol table, and the total typed projection decoder replaces a token whose output type does not match the demand at its position by that type's fallback leaf, so every gene still expresses (`papers/geneframe-gecco.tex`, abstract and Section 5; `src/geneframe.rs` per fuller `CLAUDE.md`).

This paper describes the third design built on that geneframe after symbolic regression and regular expressions: a kingdom whose genes are GPU compute kernels. It differs from the earlier two in three ways. First, the phenotype is a program with effects, not an expression: loops, barriers, stores, workgroup memory, atomics. Second, an invalid phenotype is not merely unfit; it can take the machine down. Third, fitness is measured by running the program on a device whose timings drift.

The contribution is the design that meets those three facts:

1. **A kingdom that spells the whole WGSL compute language** as typed rows of the geneframe, statements included, measured construct by construct against the W3C specification (Section 2; `docs/kingdom/WGSL_COVERAGE.md`).
2. **Safety by construction and by proof.** Duals that make a crashing or hanging kernel unrepresentable, a six-layer checker that proves what the duals cannot, and a record of the bugs the two found in each other (Section 3; `src/wgsl/check.rs`).
3. **Round trips as the correctness contract of the representation.** Text to AST to text, AST to genome to AST, AST to text to AST, each measured on 1,000 random kernels and ratcheted in tests (Section 4).
4. **Typed variation.** Knob mutation over first-class parameters, sixteen structural rewrite rules with preconditions and inverses, twelve pattern-to-action rewrites, typed subtree crossover, all through the checker (Section 5).
5. **A fitness that treats correctness as a constraint and gives infeasible individuals a graded place** through closeness columns, so that a wrong kernel is not a flat worst score (Section 6).

Results are highlights, in Appendix A, with their conditions.

Vocabulary. The paper uses the evolutionary computing terms: individual, population, fitness evaluation, constraint, objective, feasible, selection, crossover, mutation, genotype and phenotype. The code's parts are named as follows. The **parser** reads WGSL text into the kingdom's AST (`src/wgsl/safe_reader.rs`, `read_safe` at line 4492). The **emitter** writes an AST as WGSL text (`src/wgsl/render.rs`; `safe_karva::render_tree`, `src/wgsl/safe_karva.rs:801`). The **encoder** turns an AST into a Karva genome and back (`src/wgsl/safe_karva.rs`: `encode` at 191, `build` at 510). The **checker** is the layered legality proof (`src/wgsl/check.rs`). The **sampler** draws a random typed tree (`render::sample`, `src/wgsl/render.rs:2586`). The **evolution engine** is phylu's (`../phylu/src/evolve/engine.rs`), driven for this kingdom by `../phylu/examples/wgsl_fit.rs`.

---

## 2. The kingdom

### 2.1 Duals

A type in the safeWGSL kingdom is a dual `slot.form`. The slot is what the hardware stores (`f32`, `u32`, `vec4<f32>`, `mat3x3<f32>`, a register, a private array, a workgroup array); the form is what the value means and which functions are legal on it. "naga admits no implicit conversion, so a function's signature over duals is exact; the `convert` class holds the only crossings" (`kingdoms/safewgsl/TYPES.md`, opening paragraph).

The forms fall into three groups (`kingdoms/safewgsl/TYPES.md`, Forms table):

- **Value forms.** `real` (f32, f16), `int` (i32), `q15_16` fixed point and its two-limb accumulator, `index`, `count`, four code-book forms `code4`..`code1024`, `bits`, `hash32`, `hash64`, `flag`, `sign`, `opaque`. Five forms share the `u32` slot with disjoint operator sets: an `index` may be added and multiplied, `bits` may be masked and shifted, a `hash32` may be mixed; the kingdom has no row that adds a hash to an index.
- **Provenance forms.** `threadrow`, `threadcol`, `cell` ("an output cell index row*d + col owned by this thread; the only index a store to h accepts"), `wgindex` and `wgvindex` ("a workgroup-array index, masked to the array by its row"), `pindex` and `pindex16` for private arrays, `uniform` ("a u32 the same for every thread of the workgroup"), `rowbase` and `colbase` for a thread's own first row and column. The form records where the value came from, not only what it is.
- **Structure forms.** `kernel`, `grid`, `block`, `stmt`, the loop counters `counter`, `tcounter`..`tcounter4`, the bound forms `boundm`, `boundd`, `boundrows`, `boundu`, `boundx`, `lit64`, `lit4096`, the loop condition `bound` ("counter < bound, produced by nothing else"), the knob forms `wgdim`, `blk`, `unroll`, and the statement contexts `L0`..`L4` ("a statement or block inside a tile loop" at each depth) and `C0`..`C4` ("under a condition ... : no barrier").

There are 183 duals (`kingdoms/safewgsl/types.tsv`, 184 lines with the header; `TYPES.md`, "183 duals"). Each has a fallback: the value the typed projection substitutes for a codon whose dual does not match the demand. Three machine columns decide it: `fallback_bits` (the lane's 32-bit pattern, repeated across lanes), `fallback_noop` (1 only for `store.store`, which substitutes no value), `fallback_elem` (an array's fallback is its element's) (`TYPES.md`, the paragraph after the table; `types.tsv` columns 7-9).

### 2.2 Rows and signatures

A function is `class.instance` with a signature over duals; `<S.real>` stands for every real dual over scalar and vector lanes, and the loader instantiates one row per legal dual (`kingdoms/safewgsl/symbols.md`, opening). The arity ceiling is four: a node is `(fn_id, first_child, ty_code, konst)` with children contiguous in level order (`symbols.md`, Arity; the two-crate protocol, fuller `CLAUDE.md`). There are 1,529 function templates (`symbols.md`, last paragraph). By class (`kingdoms/safewgsl/functions.tsv`, column 2, counted): `reg` 341, `wg` 131, `loop` 113, `stmt` 107, `parr` 95, `block` 91, `safe` 78, `literal` 61, `subgroup` 59, `bits` 49, `shape` 44, `arith` 44, `index` 42, `atomic` 42, `convert` 38, `uniform` 36, `int` 25, `store` 20, `trig` 14, `geom` 14, `quantise` 11, `builtin` 11, `axis` 9, `load` 8, `fixed` 8, `logic` 7, `compare` 6, `bound` 6, `hash` 5, `grid` 4, `count` 4, `select` 3, `kernel` 2, `arg` 1.

The tables are generated: `gen_tables.py` writes `types.tsv`, `functions.tsv`, `TYPES.md` and `symbols.md` from one source, "so the spec and the data cannot drift" (`kingdoms/safewgsl/gen_tables.py`, header). The loader reads the TSVs (`src/wgsl/loader.rs`, per `kingdoms/safewgsl/README.md`).

Some rows are composites: small WGSL functions the kingdom defines and the emitter writes verbatim (`col_mat4x4f`, `outer_3_3`, `fx_mul`, `hash_fmix32`, `hash32_to_index`; `src/wgsl/render.rs`, the `COMPOSITE_ROWS` table ending just before line 2580). The parser reads a composite written verbatim back as its row, its body not read (commit 4ad7d93).

### 2.3 Coverage of the language

The kingdom's bar is "a fully evolvable WGSL AST ... Every WGSL compute construct must be a row of the kingdom, and every kernel naga accepts must read. A kernel our reader cannot read is a bug in our software" (`docs/kingdom/WGSL_COVERAGE.md`, Objective). The document lists the language area by area from the W3C specification, each construct with one status measured by a test kernel rather than judged from the code: `kingdoms/safewgsl/coverage_probes.py` writes one kernel per construct and the converter reads, renders and validates it.

On 2026-10-12 at fuller 0203bb7, 92 of 184 read; after thirteen fills, 181 of 184 do (`WGSL_COVERAGE.md`, Status and the list "Gaps, in the order they are filled"). The three left are naga 0.20's: `@must_use`, an `override` in `@workgroup_size`, a function-scope `const`. The fills, each its own commit with a corpus row: an f32-to-f32 conversion naga writes for an abstract literal (c0f2700); the u32 form crossings and vector bit functions (80ee1c3); `storageBarrier` as its own row and `num_workgroups` (043d494); i32 registers and operations (26ba686); `switch`, `break`, `continue`, `while`, loops starting off zero (fd776ae); swizzles, vec3 values, thread ids as vector operands, `subgroupBallot` (7bd8dca, 41f8f12); matrices (9fdba47); `modf`, `frexp`, `arrayLength` (fc1e066); module `const`, `override`, `var<private>`, `workgroupUniformLoad` (1b8ce92); helpers with early returns and pointer arguments (413f239); a storage atomic the interface declares (bb7f9aa). Workgroup shapes cover the full legal range, any x and y in 1..=1,024 with x*y <= 1,024 (6ad845b); workgroup arrays are knobs of the gene, their element type and length within 32 KB (ba6defa); integer atomics in workgroup memory (2b93f14); subgroup operations and the four subgroup built-ins (bdca29a); full unrolling as one symbol with the tile loop it unrolls (c0c1022); a tree reduction in workgroup memory as one statement row (52bcdcb).

Deliberately absent: storage-buffer atomics beyond those an interface declares, images, samplers, ray queries, binding arrays, pointers as values, non-square f16 matrices (`TYPES.md`, "Not in the table"); `atomicCompareExchangeWeak`, which naga 0.20's Metal backend does not implement (`WGSL_COVERAGE.md`, Built-in functions).

### 2.4 The interface

The kernel's bindings are a declaration, not a fixed set of names. A `Role` carries its bindings, the uniform struct's fields and their roles, the input, the output and the guard of a thread's cell; the emitter writes every name from it (commit a2280b2). A `// interface <name>` line names it; the Gram's is the default; the parser renames the kernel's input, output and uniform fields to the kingdom's roles (05b4133; `render::role_named`, `src/wgsl/render.rs:272`). Two further interfaces exist: UltraDim's cosine re-score, under which 1,000 of 1,000 random genes are legal and naga-valid (05e6079), and a storage-atomic counter (bb7f9aa).

### 2.5 The master symbol table and typed Karva genomes

The population's symbol table holds the kingdom's whole vocabulary before any kernel is added: every row as a typed function instance, every per-kernel row, the closed literal sets in full and a grid for the open ones (`../phylu` commit 589b4c08). An earlier table built from what 26 seed kernels used refused three model kernels in four; the full vocabulary adds no symbol for any of them. The table learns: a kernel declined for a literal the table lacks has it recorded in the library's `wanted_symbols.tsv` and minted at the next launch (`../phylu` dd48b328).

The genotype of an individual is one head gene, the kernel's entry `kernel.entry_safe grid body epilogue`, and homeotic tail genes holding the subtrees too large for the head. "A subtree is hoisted into a tail slot when the node it hangs from would not fit the head length; identical subtrees (by text) share one slot, so an exact repeat inside a kernel is already a shared definition. A definition references only lower slots" (`src/wgsl/safe_karva.rs:26-33`). Every gene is encoded by the generic Karva encoder at K = 4 (tail = 3·head + 1) and carries one dual code per live token in level order; a literal's payload inherits the literal's dual and an `href<t>` leaf carries slot t's root dual (`safe_karva.rs:35-39`). The population's layout in the runs reported here is head 32, tail 97, 48 slots (commit 11614a5) or 96 and 128 slots (commits c8f85b9, ceb935f).

Since 2026-10-13 every token of the gene text carries its dual, `name@T061`, the dual's stable reference in `types.tsv`; a fallback leaf is `fallback@T061` (commit ceb935f). Before that the untyped text left the projection to break a tie among a row's signatures on the first: `select(vec4<i32>(0), ..)` came back as the vec4<f32> row in 5 of 80 random rows of a phylu run.

### 2.6 The typed projection, a total decode

The way back from a genome is `build`: "a decoded, unfolded gene text projected onto the kingdom's rows under a demanded dual, so a phenotype is always a typed tree the renderer accepts; a token the demand has no row for becomes a fallback leaf, as the typed projection does on the device" (`safe_karva.rs:41-44`, `build` at 510). `conform` makes a parser's tree row-consistent, inserting count-to-index and scalar-to-vector conversions and choosing, among a name's rows under the demand, the one needing the fewest conversions (`safe_karva.rs:547`; commit c48583b). The decode is total: no gene is refused for its types. phylu's typed loader takes the fallback leaves from fuller's table, one zero leaf per value dual and none for a store (`../phylu` f644a12e, 95b2c0ee); before that every offspring touched at a block, statement or bounded-count dual was refused (108 of 108 in fit_safe18, fuller 429920f).

### 2.7 The homeotic tail and the population design

The per-chromosome tail above is the mechanism built today. In the population design a definition shared across the population is a homeotic chromosome, a row of the population table with a stable id, a root type and memoised values, executed by a level schedule, and an href names a homeotic row by its stable id, never a position (`docs/PLAN_population_dag.md`, Context and Section 1; fuller `CLAUDE.md`, geneframe paragraph). `fold_population` is the share-equivalence step over a population: every internal subtree keyed by text; a subtree in two or more individuals that fits a gene head is a shared component placed at the same reserved slot in every carrier, every occurrence a reference (`safe_karva.rs:342`; commit 7eea2d6). In `wgsl_fit` it runs after scoring each generation and rewrites a carrier only when its kernel still renders byte-equal (`../phylu/examples/wgsl_fit.rs:17-20`). Section 5.2 records what positional tail slots did to crossover and why the population design's stable ids are the answer.

---

## 3. Safety by construction

### 3.1 What legal means

"A program is legal when it parses, validates under naga, compiles on the device, terminates, writes only the output cells its thread owns, and reaches every barrier in uniform control flow. Legal does not mean correct values" (`docs/PLAN_constrained_mutation.md:63-70`). Andrew's rule of 2026-10-11 stands: a legal program that computes wrong values gets the worst fitness and dies in selection; correctness is the fitness's job. An earlier attempt to type against wrong values (a dual per uniform, a row for a register set to zero, commit 5521023) was reverted the same day for this reason (3f0200a).

### 3.2 Unrepresentable by type

The tables were made by taking the WGSL kingdom's tables and augmenting them "so a crashing or hanging kernel has no spelling" (commit 2bf6fbf; `gen_tables.py` header). The mechanisms, each a form of Section 2.1:

- **Ownership of output cells.** A store to `h` accepts only `u32.cell`, produced by `safe.cell_of` from an owned row and column (`TYPES.md`, `cell`; the Codex review notes `cell` "can only be `safe.cell_of(owned row, owned col)` or the same owned fallback", `logs/HGP/001/codex_review/overflow_decision_review.md`, row "cell collision"). A read of `h` is likewise only at the thread's own cell (`load.h_safe`); the sampler never draws the plain `load.h` (8694457; `render.rs:2590-2594`). Every store carries the bounds check of its own cell so a guard and its store cannot be built apart (694d1a6). Every kernel ends in the store of its thread's cell: `kernel.entry_safe(grid, body, epilogue)` (af092b3).
- **Thread ids are not interchangeable.** Each id is its own dual (`vec3<u32>.gid`, `.lid`, `.wid`), its x component a column-axis index and its y a row-axis index, a strided or grouped block's rows and columns their own forms. Before this, single-symbol mutants of the corpus changed a thread id or its axis in 273 of 1,000 cases and 69 of 92 engine-made offspring of fit_A2 failed correctness, typically one symbol of an index; after it, 0 of 1,000 (commit 4948c9c, body).
- **Masked indices.** A workgroup-array index is `wgindex`, "masked to the array by its row"; a private-array index `pindex`, a 16-element array's `pindex16` (4cc533c). The old `wg.get`/`wg.set` masked to 2,047 regardless of the array's declared length (a1d6614, legality item 6). The masks also let Metal drop its bounds checks: the same text masked ran in 127.8 ms against 221.6 ms unmasked at d 2,048, m 65,536 (baf4503).
- **Bounded loops.** The loop condition `bound` is "produced by nothing else" than the bounded constructors; a bound is a binding (`cfg.m`, `cfg.d`, `cfg.r_rows`), a literal at most 64 or 4,096, or a computed bound. `MAX_LOOP_TRIP = 2^24` is the kingdom's declared maximum trip count (`src/wgsl/render.rs:616`; 4ff1ac5: "not the test shapes' 65,536"); every computed bound renders `min(e, 16777216u)` (`TYPES.md`, Loop trip counts). A loop at a depth that may hold a barrier takes a uniform bound (`u32.boundu`); one bounded by a thread's value is written as conditional statements with no barrier in it (c54a302).
- **Barriers only in uniform control flow.** The conditional contexts `C0`..`C4` have no barrier row (`TYPES.md`, Forms); no `return` in a conditional context, since a later barrier would be non-uniform (d807868); a loop that breaks holds no barrier (`WGSL_COVERAGE.md`, `break`).
- **No wrap, no zero divisor.** "The kingdom cannot spell a difference that wraps: a raw difference is `u32.indexw` and a uniform difference, product or shift `u32.uniformw`, which no row takes; the kingdom's difference is the floored one (`index.sub_floor`)." A divisor that is not a literal of at least 1 renders `max(b, 1u)` (`TYPES.md`, "No wrap, no zero divisor"; commit 2555f0a). Shift counts are masked `& 31u` by the emitter (Codex review, row "Arithmetic semantics", citing `render.rs:1703`).
- **Adapter limits.** Workgroup memory within 32 KB: two f32 arrays of 2,048 and two vec4 arrays of 512 are exactly Apple's 32 KB (4d40667); a block within 64 registers (`knob.rs`, joins).

### 3.3 Proved by the checker

"Joins, patterns and table lookups choose where a rule may fire; they are not a proof" (`PLAN_constrained_mutation.md:72-76`, Codex review finding 1). The checker runs after every rewrite and names the first layer that fails (`src/wgsl/check.rs:1-5`). Its layers (`check.rs:7-14`, `Layer` at 58-66):

| layer | what is proved |
|---|---|
| 1 typed tree | every node is a row of the kingdom or the role at its signature; a loop counter read outside its loop is the constant 0 |
| 2 structural grammar | the root is the safe kernel; a store to `h`, and a read of `h`, only at the thread's owned cell; a barrier never inside an `if`; loops only from the bounded constructors, no item loop in an item loop, at most four tile loops deep |
| 3 range | every workgroup and private array index in its array's range, by its mask or by its value at the worst shape (d 8,192, m 65,536, the band 8,192 rows: `render.rs:2237-2239`); no `u32` wrap in index arithmetic anywhere |
| 4 barrier uniformity | a barrier only where every condition it depends on (an `if`, a loop's trip count, a `break` or `continue` guard, an earlier conditional `return`) is workgroup-uniform |
| 5 adapter | the workgroup's shape and its workgroup memory within the target's limits; subgroup and float features the target has |
| 6 bounded loops | every loop's trip count bounded: a literal, a binding, or a computed bound capped at `MAX_LOOP_TRIP`; no wrap in a bound's arithmetic |

Uniformity follows WGSL's rule conservatively: literals, the uniform buffer, `workgroup_id`, `num_workgroups`, the subgroup size and count, read-only storage at a uniform index, the counter of a loop whose bound is uniform, and a register every assignment of which is uniform under uniform control, by a fixpoint (`check.rs:16-23`). The range layer works in closed u128 intervals so the checker itself never wraps (`check.rs:117-125`).

The checker was built on 2026-10-12 (5b85f67: the shipped kernel and 98 corpus kernels pass, nine hand-made illegal trees fail at their layer, 80 of 1,000 sampler genes pass). The sampler and the tables were then brought to it in a day: knob references resolved before the layers (892cf77, 161 of 1,000 pass); the six legality items (0080836 a counter outside its loop is 0; 2cd3c98 a fallback cell is the thread's own; 8694457 no `load.h`; 4cc533c `pindex16`; c54a302 uniform bounds for barrier loops; a1d6614 the declared mask), after which 525 of 1,000 passed (fab27f4) and, with the wrap decision below, 1,000 of 1,000 (a8f22bc).

The adapter layer exists because of a bug the Codex review found: fuller validated with every naga capability enabled, not the device's (`PLAN_constrained_mutation.md:92-95`, finding 1). `Target` now carries the device's naga capabilities and compute limits, "from the DEVICE's features (what was requested), not the adapter's", exactly as wgpu-core 0.21's `create_validator` does; the device in use installs its target and until one does the recorded M3 Max target applies: 1,024 invocations, 1,024 per axis, 32,768 bytes of workgroup storage (`src/wgsl/target.rs:1-12`, `recorded` at 37-47; `../phylu` 14837299 installs it).

### 3.4 The u32 wrap decision and Codex's review

The decision went three ways in one day, each recorded. First, a wrap in index arithmetic was a diagnostic, not a failing layer: WGSL wraps by definition, every access is masked or owned and every loop bounded, so a wrap can only give wrong values, which selection removes (8de9dce, f361656). Second, a split: illegal in a store's cell index (a wrapped cell can be another thread's, a data race, and fitness turns non-deterministic) and in a loop bound (a wrapped bound runs to the cap and holds the GPU), a diagnostic elsewhere (d5af183, a8f22bc; `PLAN_constrained_mutation.md:83-90`).

Codex reviewed the split adversarially (`logs/HGP/001/codex_review/overflow_decision_review.md`). Its findings: the split misses non-wrapping huge uniform bounds, since `bound.x_of_uniform` was uncapped and layer 6 accepted an unknown uniform bound; the revised rule was documentation only, with range overflow still a collected diagnostic; `index.div` was a raw divide and only `band_of` guarded zero, so a zero divisor "must be rejected/proven nonzero, not treated as merely overflow"; at the declared envelope a wrapped `cell` was already unreachable (row*d + col with d = 8,192 and at most 8,192 rows is below 2^32), so the split "protects a cell case that is already unreachable ... while leaving the more credible large-uniform-loop and deployment-envelope failures open"; storage reads are safe only while wgpu's runtime bounds checks are enabled; and the checker proves only its worst envelope, so fitness cannot kill a wrap beyond tested shapes. Recommendation: "(c), illegal everywhere for index/control arithmetic, ideally made unconstructible by refined types/proofs; reject divide-by-zero and cap all loop bounds, including uniform ones. This adds no hot-loop instructions ... and the observed loss is only 47/1,000 random genes."

Third, the recommendation was adopted (2555f0a): a wrap is illegal everywhere; a raw difference and a left shift are `u32.indexw`, which no row takes, and the parser reads `x - y` floored and `x << k` as `x * 2^k`; every computed loop bound capped, uniform ones too; a non-literal divisor renders `max(b, 1u)`; a test pins wgpu's runtime bounds checks. The checker's header records the rule and where a wrap is named: `store_cell_wraps` in a cell, `loop_bound_wraps` in a bound, `index_wraps` elsewhere (`check.rs:37-48`).

### 3.5 Bugs found and fixed

The kingdom and the checker found bugs in each other, and the device found bugs in both. The record, from the commit history since 2026-10-10:

**Two offset races.** The parser accepted an offset row applied to another offset row; "nested, each offset clamped to the block, the sum left it and the cell was another thread's (the reader refused it, the checker passed it)". Fix: an offset row takes only the thread's origin, new duals `rowbase` and `colbase` with their strided and grouped kinds, and the checker's ownership proof refuses an offset of an offset (2698df7). Then a grouped offset could carry a different count of groups than its origin, which "left the thread's block, two threads' cells"; fix: a grouped offset takes its groups from its origin, in the emitter and the checker (d6e60e7).

**A sweep of 46 valid model-written kernels on the GPU** found five bugs in the kingdom: a vector local's lane keyed by the wrong name so the lane rows did not fire (71e1c6a); a `.x` store on a vector local read as a private array, which left the register untouched (c6c05bb); a continuing item loop that restarted its counter (91f39f0, 28 of 128 cases before, 128 of 128 after); a block factor of 8 snapped to 4 so a child computed the wrong cells, and vec4 workgroup arrays of the wrong length (4d40667). 45 of 46 compute as written when rebuilt; the 46th, k08, races as written (a68ef05).

**Metal translation failures.** Six random kernels of fit_SF_random failed to compile: naga 0.20 writes `-m` on a matrix to Metal as a unary minus on `metal::floatCxR`, which Metal does not define; the emitter now writes the matrix times -1.0. `degrees` and `radians` were written with naga's bare parentheses, `((x) * 57.29..)`, which Metal parses as a cast when x is a column of a matrix composed in place; the emitter writes the products. A new GPU test renders, checks and compiles the 1,000 seeded sampler trees on Metal: 1,000 of 1,000 in 915 s (58dc7e1). The ledger of that run shows the failure as it arrived: "Internal error: Metal: program_source:317:19: error: invalid argument type 'metal::float4x4' ... to unary expression" (`logs/HGP/001/fit_SF_random_stopped_emptypump/ledger.tsv`, generation 0, r1).

**An unbounded loop that held the GPU.** A decoded loop condition that differed from the seed's made an unbounded kernel that occupied the GPU until a restart, twice on 2026-10-10 (d79d6f7). This is the origin of the bounded-loop forms.

**An unfold that did not terminate.** Tail genes assembled by champion mating from two individuals read later slots twice at every slot, 2^48 nodes; one chromosome took half an hour on the CPU. `homeotic::unfold` is now bounded at 65,536 operators (495d288).

**A missing header that changed the dispatch.** The emitter's text lacked the `// grid x y bx by` line, so a 4x4 register-blocked kernel was dispatched at 1x1 blocks, sixteen times the workgroups, and ran at 0.59x the shipped kernel where its source ran at 5.4x (52ed969; also 04b89af).

**A barrier some threads skipped.** A `continue` under a non-uniform condition makes the rest of its loop body non-uniform; a barrier after it now fails layer 4 (51eb0ee).

**The continue after a break, the let that read a changed local, the loop entered off zero.** Each a parser bug by Andrew's rule that a kernel naga accepts and the parser refuses is a bug in the kingdom (ccabb66 Kahan's compensation read after its store; fd776ae; 64f8c8d and the whole series of 2026-10-11 and 2026-10-12 that took the corpus from 3 of 32 model kernels reading to all of them, 8758f73).

**A wrong value hidden by u32 wrap in an i32 literal.** An `atomic<i32>` kernel's -1 came back as 2,147,483,520 in the gene word; the literal set now holds -1024..-1 and decodes as i32 bits (`../phylu` fe0c119d).

**Every refusal is a bug.** The parser's declines were worked through one at a time until every kernel naga accepts reads: run A's 12 declines of 45 replies (64f8c8d), 91 of 94 naga-valid replies (8758f73), 433 of 441 (4948c9c), 430 of 441 with the ten left not WGSL and one flagged for a header that contradicts its own indexing (`WGSL_COVERAGE.md`, "The model replies"; 285226a). Mined from the saved replies: 535 distinct trees read, 0 parent kernels unread (`docs/MINED_REWRITES.md`, Data).

---

## 4. Round trips and representation

The kingdom has three representations of a kernel, text, AST and genome, and three programs that move between them. Their agreement is the correctness contract of the representation, measured on the same 1,000 seeded random trees and ratcheted in tests.

### 4.1 Text to AST to text: the parser and the emitter

The parser walks the entry function's statements in naga's order, inlines its `let`s, carries expressions over as typed trees "and the safe rows substituted wherever the grammar demands provenance: a load of `phi` must be `item · d + column` over the loop's own counter, a store to `h` must be read-add-write at this thread's cell, a loop must count to a binding or a small literal in steps of one, a barrier must stand outside any condition. A kernel that does not fit is refused at the first statement that fails, by name; it is never admitted by guessing" (`src/wgsl/safe_reader.rs:1-11`; the module is 5,718 lines). The emitter renders every node from its row's naga column or the composite library, and "nothing is checked twice: the text goes to naga, which validates it, and a refusal names its reason" (`src/wgsl/render.rs:1-11`).

The parser is what makes model-written and human-written kernels individuals. The measure is the corpus `tests/corpus/safewgsl` with a manifest of expected outcomes (a5a012b): 31 of 31 seed kernels and every valid model reply read (37042a4); 884 saved kernels byte-identical through parse and emit after the interface change (05b4133); 885 of 885 the same up to register names after the canonical AST (24b3345).

### 4.2 AST to text to AST: completeness and the canonical form

The completeness test draws 1,000 random trees from the sampler, renders each, passes it through the checker, parses the text and re-renders it (ee2d9ed). Its three counts, in order of strength: the tree reads back at all; the text reaches a fixpoint (emit, parse, emit gives the same text); the parsed tree equals the tree drawn. The ratchet over two days (commit bodies):

| commit | date | reads back | own text or fixpoint | AST equal |
|---|---|---|---|---|
| ee2d9ed | 10-12 | - | 108 own text | - |
| 5e67bb9 | 10-12 | 436 | 108 own, 131 fixpoint | - |
| 2698df7 | 10-12 | 538 | | |
| d6e60e7 | 10-12 | 768 | | |
| fd3ecde | 10-12 | 903 | | |
| 51eb0ee | 10-12 | 1,000 | 141 fixpoint, 110 own | |
| 58dc7e1 | 10-13 | 1,000 | 141 fixpoint, 109 own | |
| 6f21735 | 10-13 | 1,000 | | 0 (first measured) |
| 24b3345 | 10-13 | 1,000 | 563 fixpoint | 104 |
| 3334408 | 10-13 | 1,000 | 680 fixpoint | 123 |

The gap between "reads back" and "AST equal" is the subject of `src/wgsl/canonical.rs`: "The grammar spells some kernels more than one way, and the text the emitter writes cannot tell those spellings apart; the parser reads each such text as one of them. `canonical` maps any tree to that one spelling, structurally (it never emits or parses), so the sampler draws only trees a text can express and the round trip is tested against a tree, not a text" (`canonical.rs:46-53`). Its nine rules name what the text does not record: a block's grouping (nested blocks written in place, up to four statements a block); the split between body and epilogue; the grid (the dispatched grid, the workgroup plan's declarations); a literal's written value (block factors snapped, counts rounded, a knob reference as its plain literal); a fallback's zero; a reduction's count rounded to a power of two; a thread's origin written from the grid; a strided loop stepping by 1; then the parameters as the parser assigns them (`canonical.rs:55-95`). The first AST-to-AST bug found was that naga lowers `for (k = 0u; ..)` to a store before the loop and the parser read it as a register set, "so emit, parse, emit grew a kernel each pass" (6f21735). The 877 of 1,000 still unequal at 3334408 are the open work of Section 8.

### 4.3 AST to genome to AST: the encoder

The encoder's test `random_trees_survive_the_genome` encodes the 1,000 seeded sampler trees at head 128 with 128 tail slots, decodes every gene, unfolds and projects, and compares to the tree drawn: 1,000 of 1,000, 763 of them with the fallback leaf alone typed (ceb935f). Before the typed tokens the count was not 1,000: the projection broke signature ties on the first row. phylu's side of the same contract: the launch path decodes every kernel of a population as its text, 83 kernels on the host rendered equal to their texts (`../phylu` 8e00f47c); typed crossover children render their parent's text in 5 of 1,000 cases, open (`../phylu` 0e08c69e). An earlier representation bug of consequence: the engine's own random draw decoded every row to the fallback kernel because one symbol of 14,844 produces the root, so fit_random1's 80 rows and fit_RA_rules' 56 random rows were all the empty kernel; every row now comes from the kingdom's typed sampler read back through the parser (`../phylu` ac7ed361). Every random draw at launch and at a pump must express the kernel drawn; 5 of 80 did not before the typed tokens, and the count is reported on the seed and pump lines (`../phylu` 264eb51e, fa7e6778).

---

## 5. Variation operators

All operators act on the same genome. The checker stands between every operator and the GPU: a child that fails it is a line in `errors.tsv` with its origin, counted against the operator, never dispatched (`../phylu/examples/wgsl_fit.rs:35-53`).

### 5.1 Point mutation

The engine's point mutation replaces one token by another of the same type signature (a function of the same arity and duals, a terminal of the same dual, a literal of the dual with another value) in the expressed head (`../phylu` c1727a5e, 29973204). Measured on 2026-10-11: feasible on first evaluation 16 of 92 (`PLAN_constrained_mutation.md:21-27`). The reason is recorded with the axis duals of Section 3.2: a one-symbol change in a kernel is usually a catastrophic change of meaning, because a block factor appears in the grid header, the loop bounds, every stride and the store guard (`PLAN_constrained_mutation.md:29-34`). After the axis duals, 0 of 1,000 single-symbol mutants change which thread id or axis an index is built from (4948c9c). Under `--split-by-feasible` the point mutation runs at `--wild-point-rate` a slot, 0.02 by default, about 330 an offspring, on offspring of infeasible parents only (`../phylu` ca4704bc).

### 5.2 Karva crossover and the tail-slot problem of 2026-10-11

The host crossovers of `wgsl_fit`, the deduplicating recross and champion mating, swapped genes by position between two individuals. "Each kernel numbers its own tail slots, so slot k of one parent and slot k of the other are unrelated pieces, and the children lost their store" (`../phylu/src/evolve/wgsl_cross.rs:1-6`). Measured on the device over 40 parent pairs from two saved populations: positional crossover 1 and 1 of 40 children correct, typed 10 and 9 (`../phylu` 51bb500f); on the host over 1,000 pairs, all 1,000 positional children hold a fallback leaf and 24 equal parent A (`../phylu` f6b1bc6f). The same mechanism produced the 2^48-node unfold of Section 3.5. The design conclusion, from the Codex review of the plan: "shared tail definitions (positional hrefs) are unsound carriers for groups across crossover" (`PLAN_constrained_mutation.md`, Section 12, finding 2); parameters are first-class closed nodes, and lowering to shared rows comes only through the stable-row-id population design (`PLAN_constrained_mutation.md:155-166`).

### 5.3 Typed subtree crossover

A subtree of parent A is replaced by a subtree of parent B of the same dual, preferring statements, blocks and the roots of shared sections; the child is rendered and read back through the path every model kernel takes, so "the child is a well-typed kernel by construction. The duals carry the context (a statement inside the item loop, a tile loop, under a condition, the epilogue ending in the store), so a graft of the same dual keeps the kernel's structure" (`wgsl_cross.rs:6-14`). 1,000 of 1,000 typed children rebuild, validate and read, none with a fallback, 4 over the cost bound (`../phylu` f6b1bc6f). The device measurement is the one above: 10 of 40 correct. "No static interface rule separates correct children from wrong ones ... correctness is semantic, and only the device's cases decide it" (`../phylu` e598799c, body). The grid can be excluded as a crossover point (`../phylu` c23a2096).

### 5.4 The knob mutation

A knob is a declaration under the kernel root: the grid's workgroup extent and block factors (`grid.of`), the tile depth (`grid.with_ktile`), each workgroup array's length (`grid.with_wg`). A position that depends on a knob, a stride `wid.x * 64u`, a loop bound, a mask, an array's length, is a reference leaf `literal.p_<knob>[_log2|_m1]` whose payload is a multiplier of the knob in 1/256; `resolve` writes every reference as the value the kernel's declarations give, "so a change of one knob rewrites every reference in step, and a reference grafted into another kernel reads that kernel's knob. A reference names no counter, register, binding or local" (`src/wgsl/params.rs:1-16`). The parser assigns the parameters when it reads a kernel (cf05f6f); the tile depth is a knob only where the tile scales with it, found by a device measurement of 330 knob children (3297827).

The mutation picks a knob with a legal new value, a lookup in its set (powers of two within the adapter's per-axis limit; {1, 2, 4, 8}; array lengths to 8,192), and keeps it only when the joins hold: the workgroup within the adapter's invocations and axes, the block within 64 registers, the workgroup memory within the adapter's, and every reference resolving exactly into its slot's range (`src/wgsl/knob.rs:8-20`; 872072d). This is the plan's worked example, block factor 4 to 8 on the columns, as built (`PLAN_constrained_mutation.md:215-236`).

### 5.5 The structural rules

A rule is "a pattern on the typed tree with holes, the constraints under which it may fire, and a replacement that changes every coupled position together. A rule lists the sites where it fires in an individual; the operator picks one and rewrites; the child then passes the layered checker" (`src/wgsl/rules.rs:1-8`). The statement editor views a block as the flat list of its statements and rebuilds it in the grammar's spelling (`rules.rs:12-15`). The sixteen rules (`rules.rs:1811-1830`, names at 1832-1849; the last ten in `src/wgsl/rules_more.rs:1-28`):

| rule | what it changes | source |
|---|---|---|
| drop-zero-fill | removes a statement that only stores zero into private arrays or registers outside every loop, nothing touching them before (WGSL zero-initialises) | 0ce3caf |
| grow-rows | a thread's row block 4 to 8 at a constant tile: old row j is new row 2j, new row 2j+1 a copy over fresh registers; fires on contiguous and strided ownership | 14d3d45 |
| vec4-tile | a loader's strided f32 tiles stored as interleaved vec4 tiles, cell j at vec4 j % S lane j / S; the loader counts vec4s, a quarter of the passes | c126fd0, 3c00660 |
| shrink-rows | the inverse of grow-rows, 8 to 4; planned on plain values and rewritten on the kernel with its references (on the plain tree 5 of 5 children were wrong) | 0a66dc1 |
| un-unroll-k | a loop stepping two K rows a pass into one row a pass, twice the passes, the second accumulator set's statements gone | 7926296 |
| unroll-k | the inverse: an even-count loop steps two rows a pass into the same registers | 7926296 |
| drop-dead-store | every write of a register nothing reads removed (the miner's largest class) | bf43759 |
| unroll-k4 | unroll-k by four | bf43759 |
| roll-k | the inverse of unroll-k and unroll-k4, each register's terms added in the same order | bf43759 |
| split-accumulator | odd K rows into a fresh partner zeroed before the loop and added after it | bf43759 |
| wg-shape | the workgroup reshaped at the same thread count (16x16, 32x8, 8x32), only where no plain index literal encodes the workgroup | bf43759 |
| hoist-load | a value read from memory that does not change in a loop read once before it | bf43759 |
| pad-shared | a workgroup array padded by one element a row: index x read at x + x / W, a bound the module proves keeps every index below the length | bf43759 |
| grow-cols | the column analogue of grow-rows: a thread's columns 4..7 at the column tile's 2x+1 | bf43759 |
| shrink-cols | the inverse of grow-cols | bf43759 |
| double-buffer | the loader one pass ahead into the other half of each tile, one barrier a pass instead of two; statements moved out of the loop re-typed by their rows' signatures | bf43759 |

Measured on the corpus and the saved population popR_rules, every child of every rule passes the checker: drop-dead-store 581, pad-shared 165, double-buffer 104, wg-shape 70, split-accumulator 56, unroll-k4 56, grow-cols 24, hoist-load 4; unroll by 2 or 4 then roll gives back the kernel's text (112), grow then shrink the columns too (24) (bf43759, body). On the device, un-unroll-k and unroll-k children are correct and the two directions time as expected (7926296); shrink-rows gives 5 of 5 correct children, slower than their parents, median 0.77 to 1.01 (0a66dc1). Over 49 kernels of popR_rules, grow-rows fires on 24, vec4-tile on 25, none on both, and after one climbing step no climbing rule fires again (3423d77). Not built: symmetric-half, because a thread may store only its own cell of h and the fitness dispatches h in row bands, so a mirrored write is outside both (`rules_more.rs:28-30`).

### 5.6 The generic typed rewrites

"A mutation is pattern -> action, like awk. Each rewrite walks the tree, collects the nodes its pattern matches, picks one with the run's RNG and applies its action. The patterns are typed by the kingdom's dual table only: a rewrite knows nothing of what the kernel computes, nor whether it is correct, so it applies in every kingdom" (`src/wgsl/rewrites.rs:1-7`). The twelve (`rewrites.rs:18-31`, names 49-62): swap-operands (a binary node's two same-dual children), swap-leaves (two same-dual variable leaves), index-literal (an index literal moved by 1 or 2), replace-leaf (another terminal row of the dual), swap-op (another row of the same signature), delete-statement, duplicate-statement, swap-adjacent, wrap-unary and unwrap-unary (a unary row from a dual to itself), loop-end (a counted loop's trip count by 1), insert-accumulate (`r = r + a * b` after a statement, every row found in the dual table) (23ea7d4, 726962b). All twelve match the corpus and their children check. The rewrites and the rules make one pool; `rewrites::mutate` draws among every operator whose pattern matches the individual (`rewrites.rs:7-10`).

An earlier module, `fix_edits.rs`, routed correctness edits by the parent's closeness columns; it was removed on 2026-10-13 as "routed by correctness and specific to the Gram" (23ea7d4): no routing by correctness, selection decides (`../phylu` 8f9bcaa5).

### 5.7 The engine: islands, pump, cohorts, random immigrants

`wgsl_fit` hands phylu's engine an external fitness, and "everything else is `Engine::fit` as every kingdom runs it: the intake and champion islands, the pump stamping cohorts, virtual ALPS, elitism, the device variation, the genealogy, the telemetry stream" (`wgsl_fit.rs:1-8`; `../phylu` d6ac88dd). The engine's parameters as the run of Appendix A set them: intake island 64, champion island 16, elites 2, tournament fraction 0.07, pump every generation, cohort merge 20, arrival children 4, promote fraction 0.03 (`logs/HGP/001/fit_RA_rules/stream.jsonl`, `run_start`; `../phylu/src/evolve/engine.rs:1086-1090`, 1115, 1176, 1207, 1269). Virtual ALPS is the cohort preference: a pump beat stamps arrivals with a cohort, and tournaments prefer same-cohort pairs until the cohorts merge (`engine.rs:1151-1155`, 1263-1269; `papers/geneframe-gecco.tex`, Section 7.2).

The pump's refill rows are the random immigrants. With `--rule-all`, every rule at every site over every distinct feasible member of the generation just scored enters the refill, best parents first, one child per parent a round (`wgsl_fit.rs:55-62`; `../phylu` 3afcb19e). With `--pump-random` or `--seed-from none`, the refill rows are new random kernels of the typed sampler, none a kernel the run has measured (`../phylu` fa7e6778). Deduplication is by kernel text: a genome whose kernel was measured before never reaches the GPU (`wgsl_fit.rs:16-17`; `../phylu` 29973204).

---

## 6. Fitness

### 6.1 The problem

The challenge kernel is the weighted Gram `H = ΦᵀΦ`, built through the same banded driver shape as the production code: item bands and output-row bands sized from the per-binding limit, the accumulator resident between bands (`src/wgsl/gram.rs:1-13`). The reference is `H` in f64 on the host; the criterion is `|H_gpu − H_ref| ≤ τ ‖φ_·r‖ ‖φ_·c‖` with `τ = 4 ε_f32 √m` (`gram.rs:14-16`).

### 6.2 The correctness constraint

The constraint is 128 stratified test cases over eleven input kinds, each chosen to expose a class of wrong kernel: unit Gaussian rows; weighted rows over six decades; low rank plus noise; 90 % and 99 % sparse; heavy-tailed Student t; cancelling pairs (each row with its negation, so every off-diagonal cell is an exact doubling); small integers, where every sum is exact; extreme magnitudes 1e-20 and 1e15; NaN and infinities at known cells; duplicate and constant rows; ragged shapes (`gram.rs:95-130`). The 128 are resident on the device and run 16 to a submission (40ea250). A feasible individual also passes the reference pairs: a few training, validation and extrapolation pairs with an f64 reference and the shipped kernel's own largest ratio on each (`gram.rs:985-1000`). The failure detail reads "constraint N/128 cases: ..." and names the case (13a47d0).

### 6.3 Timing with paired re-timing and drift

Timing is paired: each candidate build is timed against the shipped kernel on three problem shapes, the speed-up is the median over `timing_rounds` alternating rounds, and "one round scattered equal kernels by 15 % (the re-timing of 2026-10-11); five or more are needed to rank kernels" (`gram.rs:1009-1011`; 383489d). The shipped kernel is timed once a generation, not beside every candidate; speed-ups divide by the least shipped time so far, and a shipped measurement more than 10 % above the least is flagged as contended (daea24f, 3f31d04; `gram.rs:1371-1377`; `../phylu` be4f97c2). Elites are re-timed once a generation (`../phylu` 83927cfb). The drift is physical: a laptop GPU lowers its clock under sustained load, which scales both kernels alike (acd0964), and the shipped kernel's times on the generation lines of a run stay within 1 % (1,004-1,016 ms across six generations, `logs/HGP/001/fit_RA_rules.log`). The re-timing that motivated five rounds showed four top kernels from HFF 0.3300 to 0.3204 running within 1 % of one another where a single paired round had spread them from 6.0x to 8.7x (82185e3, 0155241).

### 6.4 The HFF angle

The fitness is six columns in [0, 1] with 0 best: the inverse speed-up on each of three shapes, clamped to [0, 1], and the worst error relative to the shipped kernel's on the training, validation and extrapolation pairs; the angle is `θ = acos(1 − mean(x²))`, phylu's `hff.wgsl` lines 212-213 (`gram.rs:985-993`, `fitness_columns` at 1360). The generation-0 record of the first run had a speed-up ranking only, which was wrong; this is the fix (933a0d8). Validation inside the angle is why no parsimony term is needed; the fixed head bounds size (user memory, "Validation in HFF replaces parsimony").

### 6.5 The graded place of an infeasible individual

A wrong kernel is not a flat worst score. Ten closeness columns in [0, 1], 0 the reference itself, are computed on the host from the outputs the constraint already reads back: cases failed, Pearson correlation, scale, shift by one or two rows or columns, transpose, sorted values, row and column sums, cells written, cells within tolerance, input sensitivity ("a kernel blind to Φ") (`src/wgsl/closeness.rs:1-24`, `COLUMNS` at 29). The `scale` column is logged only: it pointed the wrong way on all three saved runs, AUC 0.32, 0.22, 0.25 (`closeness.rs:183`; 7d834fd). The infeasible band is `[π/2 + 0.001, π/2 + 1)`, strictly worse than every feasible fitness, the angle of the nine columns mapped into it (`wgsl_fit.rs:116-135`). Before the band, one flat π/2 "so selection could not tell 34 of 128 cases from 0" (`wgsl_fit.rs:120-121`). The ledger carries the columns per evaluation (`ledger.tsv` header: `cases distance closeness`).

### 6.6 The static cost bound and the watchdog

Work the GPU has started cannot be cancelled, so a kernel whose static cost exceeds 64x the shipped kernel's per-thread cost per item is refused by name before dispatch (5a01100; `safe_karva::rebuild_bounded`, `safe_karva.rs:850`). An expression-bounded item loop is costed by its passes at the worst shape (ee4dd07). The fitness measures each kernel's time and stops a slow one before a large shape; the static estimate only guards (66e16aa). A watchdog thread exits the process when an evaluation exceeds its budget, naming `current.wgsl` (`../phylu` da2636cd, df9cd01b). A run is stopped by deleting its run file, never by a signal: a kill with a command buffer on the device wedged the GPU (`wgsl_fit.rs:86-89`).

---

## 7. Where it lives

| part | file |
|---|---|
| tables | `kingdoms/safewgsl/{types,functions}.tsv`, generated by `gen_tables.py` |
| loader | `src/wgsl/loader.rs` |
| parser | `src/wgsl/safe_reader.rs` |
| emitter | `src/wgsl/render.rs`, `safe_karva::render_tree` |
| encoder | `src/wgsl/safe_karva.rs` |
| canonical form | `src/wgsl/canonical.rs` |
| checker | `src/wgsl/check.rs`; device target `src/wgsl/target.rs` |
| sampler | `render::sample` |
| parameters, knob mutation | `src/wgsl/params.rs`, `src/wgsl/knob.rs` |
| structural rules | `src/wgsl/rules.rs`, `src/wgsl/rules_more.rs` |
| generic rewrites | `src/wgsl/rewrites.rs` |
| rule mining | `src/wgsl/tree_diff.rs`, `docs/MINED_REWRITES.md` |
| closeness | `src/wgsl/closeness.rs` |
| fitness | `src/wgsl/gram.rs` |
| engine and operators | `../phylu/src/evolve/{engine,wgsl_cross,wgsl_model,wgsl_population,wgsl_sample}.rs`, `../phylu/examples/wgsl_fit.rs` |

phylu depends on fuller; fuller never depends on phylu (fuller `CLAUDE.md`).

---

## 8. Limitations and future work

**The search space.** 183 duals and 1,529 templates, instantiated per dual into the loader's rows (908 at 4948c9c, 1,454 at 892cf77), over a head of 32 to 128 and up to 128 tail slots: the random-start runs of Appendix A.2 show that a random kernel of this space is almost never a correct Gram kernel, and 65 generations of point mutation and typed crossover did not make one. The structural rules climb from a correct kernel; they do not find one. What finds one today is a model-written or human-written seed read by the parser.

**The round trip is not yet an identity.** 123 of 1,000 random trees return to themselves through emit and parse (3334408). The remaining differences are spellings the text does not record; each group found is a canonical rule or a parser fix, and the test ratchets. Until the count is 1,000, a rule child's genome is re-read from its text and may be laid out differently from its parent's.

**Beam search and auto-tuning by the structural rules.** The rules are closed under inversion (unroll then roll, grow then shrink give back the text, bf43759) and after one climbing step no climbing rule fires again (3423d77), so the rule graph over a kernel is small and enumerable. The plan's auto-tuning section keys a variant table by problem shape and keeps the fastest variant per key, with the evolved population as the tuner (`PLAN_constrained_mutation.md:386-396`). `--rule-all` already enumerates every rule child of every feasible member each generation; a beam over several steps, selection ranking the whole frontier, is the next step. Open: whether meaning-preserving rules may bypass the correctness cases with a sampled check (`PLAN_constrained_mutation.md`, Open questions 2).

**Symmetric-half.** Computing the upper triangle and mirroring is not a kernel rewrite: a thread writing its mirror cell breaks own-cell ownership. It is a dispatch-plan change, an upper-triangle grid plus a mirror pass, with its own ownership definition (`PLAN_constrained_mutation.md`, Section 5.2 table; `rules_more.rs:28-30`). The dispatch plan slot of the gene root is designed and not built (`PLAN_constrained_mutation.md:178-186`).

**Mined rewrites.** 690 model replies, 535 distinct trees, 108 pairs in which a feasible child beats its parent, 3,785 edits, a median of 25 edits per pair (`docs/MINED_REWRITES.md`, Data). The edits are classed, generalised by anti-unification into 235 candidate rules with support from two children or more, ranked by mean gain (`MINED_REWRITES.md`, Candidate rules; `src/wgsl/tree_diff.rs:1-30`). Two of the top candidates are already rules (the register-block change is grow-rows and shrink-rows; the dead register initialisation is drop-dead-store). "No pair here has edits of one class only", so the gains are confounded; the clean-example method, one change per reply, is designed and not run (`MINED_REWRITES.md`, Edit classes; `PLAN_constrained_mutation.md:373-378`). None of the mined rules is wired into evolution.

**The interface.** Three interfaces exist (Gram, cosine re-score, counter); textures and samplers are not built (`WGSL_COVERAGE.md`, closing). A kernel with a binding no declared interface holds needs that interface declared first.

**Determinism of timing.** The fitness's timing columns are a physical measurement with a 15 % single-round scatter; five rounds are the working minimum. A run on a laptop GPU shares the device with the window server; the hard reset of 2026-10-08 under two builds and two GPU suites is the reason one process at a time is the rule (user memory, "Never two cold builds").

**Shared definitions across the population.** The per-chromosome homeotic tail is built; the population-wide homeotic rows with stable ids, the level schedule and the device arena are designed and measured on the host, not built into the generation loop (`docs/PLAN_population_dag.md`; fuller `CLAUDE.md`, In flight).

---

## Appendix A. Highlights

Conditions common to all runs: Apple M3 Max, Metal, wgpu 0.20.1, naga 0.20 (`target.rs:37-38`); the Gram problem of Section 6.1; the shipped kernel at about 1,010 ms per build at the timing shapes (`fit_RA_rules.log`, generation lines); one GPU process at a time.

### A.1 The rule-driven run, fit_RA_rules

Run `wgsl_fit_safe_3`, 2026-10-12 19:22Z to 19:33Z, seed 3, population 80 (intake 64, champion 16), 24 model-written seed kernels, pump every generation, 20 generations budgeted, stopped by removing the run file during generation 6 (`logs/HGP/001/fit_RA_rules/stream.jsonl`, `run_start`; `fit_RA_rules.log`, "killme.trigger removed"). Every rule child of every feasible member entered the pump each generation (`--rule-all`).

Best feasible individual per generation (`ledger.tsv`, lowest HFF among feasible rows; speed-ups at the three shapes):

| gen | row | HFF | speed-up x3 | origin | change |
|---|---|---|---|---|---|
| 0 | r7 | 0.1192 | 9.33 / 9.60 / 9.66 | seed | |
| 1 | r54 | 0.1207 | 9.29 / 9.37 / 9.47 | dedup | |
| 2 | r55 | 0.1108 | 10.14 / 10.32 / 10.66 | rule grow-rows of r43 (0.1254) | rows 4 -> 8 over wg.vb, 16 storages |
| 3 | r22 | 0.1058 | 10.67 / 11.18 / 11.25 | rule unroll-k of r24 (0.1137) | K loop of 8 statements |
| 4 | r31 | 0.1052 | 10.95 / 11.14 / 11.28 | dedup of r31, r30 | |
| 5 | r49 | 0.1053 | 10.87 / 11.29 / 11.14 | rule unroll-k of r34 (0.1125) | K loop of 8 statements |

The lineage of the generation-3 best, three rule steps (`ledger.tsv`, rows r51, r43, r55, r24, r22): r1 (0.1199) -> un-unroll-k -> r51 (gen 1, 0.1236) -> dedup -> r43 (gen 2, 0.1254) -> grow-rows -> r55 (gen 2, 0.1108) -> dedup -> r24 (gen 3, 0.1137) -> unroll-k -> r22 (gen 3, 0.1058). The generation-4 best, r31 (0.1052), is a dedup child of r31 and r30, each an unroll-k child of generation 3 (0.1061, 0.1066).

Operators by outcome over the run (`fit_RA_rules.log`, "by origin"): rule all 180/180 feasible; seed 24/25; dedup 72/312; mutation 13/83; typed cross 8/25; champion mating 3/18; pump 0/75. By rule (`ledger.tsv`, column `rule`): un-unroll-k 55, shrink-rows 53, unroll-k 38, grow-rows 33, vec4-tile 1, all feasible. 560 individuals evaluated in 714 s (`fit_RA_rules.log`, "done").

**Paired re-timing.** The ledger's speed-ups are single rounds taken as the run went, and the GPU drifts (Section 6.3). After the run the generation-0 best (`gen0_r7`) and the final best (`best_g004_r31`) were re-timed against the shipped kernel in six interleaved rounds at the fitness's three shapes, with the correctness constraint re-run each round (`logs/HGP/001/fit_RA_rules/retime_paired/gate.log`, `verdicts.tsv`, `run.out`: 18 evaluations, 20:34-20:35Z, 2026-10-12; shipped measured 1,012 / 1,006 / 1,013 ms). Per round, the speed-up at the three shapes (`verdicts.tsv`, last column; measured ms from `gate.log`):

| round | g004_r31 speed-up | g004_r31 ms | g000_r7 speed-up | g000_r7 ms | shipped ms |
|---|---|---|---|---|---|
| 1 | 10.86 / 11.03 / 11.00 | 97 / 95 / 100 | 9.32 / 9.58 / 9.37 | 111 / 108 / 117 | 1013 / 1009 / 1018 |
| 2 | 10.80 / 11.03 / 10.91 | 97 / 94 / 100 | 9.30 / 9.50 / 9.49 | 111 / 109 / 117 | 1016 / 1012 / 1020 |
| 3 | 10.76 / 10.91 / 10.42 | 97 / 95 / 104 | 9.02 / 9.06 / 8.92 | 113 / 113 / 121 | 1017 / 1014 / 1019 |
| 4 | 10.30 / 9.49 / 9.26 | 101 / 109 / 118 | 8.73 / 7.82 / 7.68 | 119 / 131 / 141 | 1099 / 1107 / 1110 |
| 5 | 9.98 / 8.91 / 8.57 | 104 / 117 / 126 | 8.41 / 7.31 / 7.24 | 123 / 141 / 148 | 1131 / 1129 / 1128 |
| 6 | 10.05 / 8.95 / 8.66 | 104 / 117 / 127 | 8.47 / 7.40 / 7.25 | 123 / 138 / 148 | 1126 / 1124 / 1122 |

The medians over the six rounds, the fitness's own rule (`gram.rs:1009`): g004_r31 10.53 / 10.20 / 9.84x at 99 / 102 / 111 ms; g000_r7 8.88 / 8.44 / 8.30x at 116 / 122 / 131 ms. g004_r31 was the faster kernel in 6 of 6 rounds at every shape. The drift is visible in the table: from round 4 the shipped kernel itself slows by about 10 % and both candidates with it, which is why the paired figures, not the ledger's single rounds, are the ones to quote. Rounded: the rules took the best kernel from about 8.5x to about 10.4x the shipped kernel.

### A.2 The random-start runs

Two runs seeded entirely from the typed sampler (`--seed-from none`), every model off, population 80, pump every five generations.

**fit_SF_random** (`logs/HGP/001/fit_SF_random_stopped_emptypump`): 65 generations, 5,280 individuals, 2,772 s, 0 feasible (`fit_SF_random.log`, "done"). By origin: dedup 0/735, mutation 0/3,408, pump 0/612, seed 0/52, typed cross 0/781. Stages reached (`ledger.tsv`): 3,735 failed the constraint, 7 failed to compile (the Metal matrix negation of Section 3.5), 1 failed to rebuild. The best closeness angle among infeasible individuals fell from 1.2573 at generation 0 to 1.2209 at generation 32 and did not move after (`generations.tsv`). The pump's second pass evaluated 0 individuals at every beat: with every model off and no feasible parent the pump had no source (`generations.tsv`, pass 2 lines; `../phylu` fa7e6778, which added random refill rows for this reason).

**fit_SF_pump** (`logs/HGP/001/fit_SF_pump_paused_g59`): 59 generations, 4,800 individuals, 2,948 s, 0 feasible (`fit_SF_pump.log`, "done"). By origin: dedup 0/2,428, mutation 0/1,392, pump 0/102, seed 0/53, typed cross 0/668. Stages: 4,206 constraint, 11 cost bound, 1 rebuild. The best closeness angle fell from 1.2954 to 1.2719 by generation 13, held to generation 40, dropped to 1.2395 at the pump of generation 40 (its first pass with random refills, 51 evaluated) and to 1.2247 by generation 57 (`generations.tsv`).

What was found: the closeness band orders random kernels and the ordering moves under selection; two Metal compile failures and one parser bug (33 of 130 random kernels did not read back on a bound of 0, e8e697f) were found by these runs. What was not found: a feasible kernel. The runs are the measurement of the search-space limitation of Section 8.

### A.3 Measurements of the representation

| measure | value | source |
|---|---|---|
| duals; function templates | 183; 1,529 | `types.tsv`; `symbols.md` |
| WGSL construct test kernels that read, render, validate | 181 of 184 | `WGSL_COVERAGE.md` |
| random sampler trees that render, check and compile on Metal | 1,000 of 1,000, 915 s | 58dc7e1 |
| random trees that read back | 1,000 of 1,000 | 51eb0ee |
| random trees to a text fixpoint; to the identical AST | 680; 123 of 1,000 | 3334408 |
| random trees through genome and back | 1,000 of 1,000 | ceb935f |
| random genes legal under the checker | 1,000 of 1,000 | a8f22bc |
| saved kernels byte-identical through parse and emit | 884 of 884 | 05b4133 |
| single-symbol mutants changing a thread id or axis, before and after | 273 then 0 of 1,000 | 4948c9c |
| model-written seed kernels reading | 31 of 31 | 37042a4 |
| model replies reading | 430 of 441 | `WGSL_COVERAGE.md` |
| rule children through the checker (eight rules) | 1,060 of 1,060 | bf43759 |
| positional and typed crossover, correct children on the device | 1 and 10 of 40 | `../phylu` 51bb500f |

### A.4 The model-seeded baseline before the rules

Before the rules, with the model operators on, the hall of fame of 2026-10-11 held four correct kernels 6.5-6.9x the shipped kernel at the fitness's three shapes, re-timed seven rounds each (643da63), and the last native-operator run found nothing better than its seeds in 1,040 evaluations over 50 generations (`PLAN_constrained_mutation.md:16-20`). The rule-driven run of A.1 improved its best in four generations.

---

## Appendix B. Figures and tables the paper should have

| # | figure or table | data source |
|---|---|---|
| F1 | The three representations and the five programs between them (text, AST, genome; parser, emitter, encoder, decoder, checker), with the device at the end | Section 1 naming; `src/wgsl/mod.rs` |
| F2 | A kernel as a typed tree: the shipped Gram kernel's AST with duals on every node, the head gene and its tail slots | `safe_karva.rs` test "the shipped kernel both ways" (11614a5); `logs/HGP/001/gen0/karva` |
| T1 | The forms table: value, provenance and structure forms with their legal slots | `kingdoms/safewgsl/TYPES.md`, Forms |
| T2 | Rows by class | `functions.tsv`, column 2 (Section 2.2) |
| T3 | WGSL coverage by area: covered, naga 0.20, binding, not compute | `docs/kingdom/WGSL_COVERAGE.md` |
| T4 | The checker's six layers | `src/wgsl/check.rs:7-14` |
| F3 | The wrap decision's three states in one day and Codex's findings | commits 8de9dce, f361656, d5af183, a8f22bc, 2555f0a; `codex_review/overflow_decision_review.md` |
| T5 | Bugs found by the device, the checker and the parser in each other | Section 3.5, the commits listed |
| F4 | The completeness ratchet: reads back, fixpoint, AST equal over the commits of 10-12 and 10-13 | Section 4.2 table |
| T6 | The nine canonical rules | `src/wgsl/canonical.rs:55-95` |
| T7 | The sixteen structural rules with what each changes and its inverse | `rules.rs:1811-1849`; Section 5.5 |
| T8 | The twelve generic rewrites | `rewrites.rs:18-62` |
| F5 | The knob mechanism: one declaration, every reference rewritten in step | `params.rs:1-16`; `PLAN_constrained_mutation.md:215-236` |
| F6 | Positional against typed crossover: fallback leaves and device-correct children | `../phylu` f6b1bc6f, 51bb500f |
| T9 | The 128-case constraint by input kind | `gram.rs:95-130` |
| T10 | The ten closeness columns and what each catches | `closeness.rs:1-24` |
| F7 | fit_RA_rules: best HFF and speed-up per generation, the three-step lineage drawn | `fit_RA_rules/ledger.tsv`, `genealogy.tsv` |
| F8 | fit_SF_random and fit_SF_pump: best closeness angle per generation, pump beats marked | the two `generations.tsv` |
| T11 | Operators by outcome in fit_RA_rules | `fit_RA_rules.log`, "by origin"; `ledger.tsv` |
| T12 | Mined edit classes with support and mean gain; the top candidate rules | `docs/MINED_REWRITES.md` |
| F9 | The population design: ordinary and homeotic rows, hrefs by stable id, the level schedule | `docs/PLAN_population_dag.md`, Section 1 |
