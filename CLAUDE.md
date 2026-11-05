# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## FULLER DOES NOT EVOLVE CODE (Andrew, 2026-11-03)

**fuller is the symbolic side only: kingdoms (the kingdom warehouse and its views), the AST, gene encoding and
decoding, program rendering (Futhark, WGSL, Python), and symbolic rewriting (the e-graph, snap, fold). It does NOT run
evolution and does NOT measure fitness.** phylu (`../phylu`) owns every genetic operation and every fitness
measurement: the generation loop (virtual ALPS, the pump, the islands), variation, selection, the population, the HFF
schemes, GPU scoring, calling external compilers and programs (`evolve::program_eval`), the runners (Andrew's
architecture diagram, 2026-11-04). Working from this directory is no reason to put code here. Never add a generation
loop, a population, selection, a fitness evaluator or a runner to fuller; phylu calls fuller to decode, render and
rewrite. Code still here that the diagram puts in phylu (`fitness`, `telemetry`, `run_file`, `population_dag`) moves
when fuller and phylu merge into one repository: the plan is `docs/PLAN_futhark_kingdom.md` 6.2.1, reviewed by
Codex (6.2.2).

## What this is

fuller is a Rust crate (PyO3-exposed as the Python module `fuller`, imported by hff) for symbolic programs: the gene
frame's kingdoms, decoding genes into programs, rendering them, and rewriting them with **egglog 2.0** (our fork in
`vendor-egglog`). It replaced sympy for the SR engine in `/Users/andrewmorgan/Dev/gamakon/hff/`: deterministic,
real-domain, bounded.

## Build & test

```bash
cargo test                       # Rust tests (plain build, no PyO3 link needed)
cargo test --features wgsl       # with the WGSL kingdom
cargo clippy --all-targets       # MUST be clean, with and without --features wgsl
maturin develop --release        # build + install the Python extension (fuller)
```

**Hard rule from the user: zero warnings, none hidden.** Never silence with `_`-prefix or `#[allow(...)]` — fix the
root cause. `RUSTFLAGS="-D warnings"` build and clippy clean in every feature set (fuller: none, `wgsl`; phylu: none,
`gpu`, `gpu,wgsl`) before any push. Tests must be fast (fail fast); a long verification test is not added during
development. Before main: both repositories' full suites plus the GPU tests.

## The gene frame: one fixed schema, one kingdom warehouse

- **The schema** (`docs/DESIGN_geneframe.md`, the live design): a FIXED abstract data model. Kingdoms are configured
  by rows only, never by changing the data model. Four row kinds: **Argument** (a function's typed inputs and output),
  **Flow** (properties computed up or down the tree: sizes, depth, order, bindings, environment), **Guard** (a
  condition with strength `must`, repaired in the decoder, or `advise`, charged in the cost) and **Entry** (the root
  and its parameters). **Types are data types** (the data types of function arguments; dozens, never thousands).
  **Sizes and other facts are properties** carried with values. **Scope is context**: a generic `var.T` with an index
  word, resolved at decode (de Bruijn, innermost binder first). A closed formula language (`src/geneframe/formula.rs`).
  Every gene decodes into a legal program by construction.
- **The kingdom warehouse** (`kingdoms/master/`): ONE set of tables for every kingdom. `kingdoms.tsv` lists the
  kingdoms and variants (`parent.name`); `data_types`, `symbols`, `spellings`, `arguments`, `flows`, `guards`,
  `repairs`, `entries` each carry a leading `kingdom` column; `properties.tsv` is the shared property catalogue.
  Loading a kingdom selects it and filters (`src/geneframe/master.rs`: `master::select("tsr.order")`,
  `master::view("futhark", "kmeans")`). ONE writer, `kingdoms/master/write_rows.py`: each kingdom's generator
  replaces only its own rows; `--check` is a test; export/import of a kingdom to JSON and back. Never hand-edit the
  tables: change the generator.
- **The kingdoms** (all on the warehouse): `futhark` (signature views a7, histogram, kmeans, kmeans_i32, pathfinder,
  reduce, scan; the pixel family `futhark.pixel.*`), `sr` (symbolic regression, a depth property), `tsr` (Typed
  Symbolic Regression: depth and Horton-Strahler stream order as properties; variants single, order, order_depth,
  order_depth_windows), `regex` (POSIX, plus the counted-repetition guard), `wgsl` (variants `wgsl.gram` and
  `wgsl.image`, nine data types). **TSR is Typed Symbolic Regression**, never "Transcendental".
- `src/typed_environment.rs` builds a kingdom's typed environment from the rows; `src/typed_gep.rs` the typed Karva
  gene (per-type head and tail; master head h = 1 + (N-1)n_max, tail t = h(n_max-1) + 1; Dc lanes hold constants, Dv
  lanes hold `var.T` index words). `src/orf_padding.rs`: the champion-pool overlay and frequency draws for the unread
  cells of a gene (design section 11).
- Per-kingdom loaders: `src/futhark_table/kingdom6.rs` + `src/futhark_view/master6.rs` (Futhark),
  `src/futhark_pixel/rows.rs` (pixel), `src/tsr_kingdom/rows.rs` (SR and TSR, with the device rules phylu decodes on
  the GPU and their host twin), `src/regex_typed.rs` (regex), `src/wgsl_rows.rs` (WGSL, with the share pass).

## Other modules

- `src/expr.rs` — `Math` datatype (real-domain ops incl. `Protected*`) + `GUARD_RELATIONS`.
- `src/eval.rs` — real-domain evaluator over egglog `Term`s (`sqrt(neg)/log(<=0)/div0 -> NaN`; protected ops match the
  engine's semantics).
- `src/karva.rs` — Karva <-> `Math`, keyed on `semantic_id`; `master_pset()`.
- `src/ruleset/` — egglog rulesets: `identities`, `powers`, `distribute`, `rational`, `trig`.
- `src/extract.rs` — `denoise()`: bounded saturation (algebra+powers) -> variants -> scored -> smallest within R²
  tolerance. Never raises.
- `src/physics.rs` — `generate()`: physics-prior candidate generator (no scoring; `speculative` tags).
- `src/snap.rs`, `src/snap_karva.rs` — constant snapping (π, e, √2, G ...).
- `src/saturated_fold.rs`, `src/population_fold.rs`, `src/population_dag.rs` — fold and the population DAG (the DAG
  moves to phylu at the merge).
- **Futhark**: `src/futhark_reader.rs` (Futhark text -> gene), `src/futhark_sig/` (the interpreter, `interp6`),
  `src/futhark_saturation/` (the e-graph, generated from the rows), `src/futhark_cost.rs` (the static cost, a
  polynomial in the dimension roles), `src/futhark_exec.rs` (compiling with Futhark's own pass list and a capped
  fusion loop). The product decision and the measured A7 result: `docs/PLAN_futhark_product.md`,
  `docs/measurements/`. The evaluator and the GPU cost are phylu's (`src/futhark_eval/`).
- **WGSL** (feature `wgsl`): `src/wgsl/` (naga front end and target, the image interface, the cuneus originals, the
  oracle, snap, the work guard). Reference programs as WGSL text in `kingdoms/wgsl_true/gram/`. The WGSL reader
  (text -> gene) and the WGSL optimisers are to be rebuilt over the rows (plan 7.7).
- `src/regex/` — the regex AST, parser and rules.
- `src/parity.rs` + `src/bin/parity.rs` — SymPy-parity scorer, per family.
- `src/python.rs` — PyO3: `denoise`, `denoise_karva`, `physics_mutate`, `physics_mutate_karva`, `master_pset`.
- `docs/index.html` — the documentation index of this repository and phylu, GENERATED by `docs/build_index.py`
  (default `--check`; a Rust test runs it and walks every link). Edit the generator, not the page.
- Compiler: Futhark's `c` backend is Homebrew Futhark 0.28.1; the GPU path uses our rust-wgpu fork
  (`futhark-rust-backend/`, patches 0001–0016; `futhark-rust-rt/`, `futhark-wgpu-shim/`). A compiler fault gets its
  note in `docs/issues/` first.

## Every evolution runner meets one standard

Every evolution loop (all in phylu) writes `OUT/stream.jsonl` through `fuller::telemetry`, honours
`OUT/killme.trigger` (`fuller::run_file`), and emits a mutation event for every operator outside plain GEP variation
(`docs/STANDARD_runners.md`). **Watch any run with ONE app, `evowatch`** (phylu: `cargo build --release --bin
evowatch`; `evowatch OUT`, `--web 8787`, `--compare OUT2`; `../phylu/docs/WATCHER.md`). Give Andrew the `evowatch`
command line at every run launch.

## Fitness is a named scheme, chosen per experiment

`fuller::fitness::SCHEMES` holds every fitness function a runner offers (moves to phylu at the merge). A runner takes
`--fitness NAME`, defaults to its family's newest scheme and records it in its header and the stream's `run_start`. A
new or fixed fitness is a NEW scheme; never change what an existing name computes; old schemes stay, marked NOT TO BE
RUN AS IS with their fault (`docs/FITNESS_SCHEMES.md`).

## Non-obvious things that will bite you

- **Rule families are NON-CONFLUENT.** distribute + trig (or + rational) co-saturated explode the e-graph. Keep them
  in separate `Family`s; `denoise` uses only bounded algebra+powers. **Always kill-guard a saturation/parity run**:
  `( cmd & PID=$!; for i in $(seq 1 N); do kill -0 $PID 2>/dev/null||break; sleep 1; done; kill -9 $PID 2>/dev/null )`.
- **distribute must NOT go in `denoise`** — `extract_variants` over its unbounded class hangs. Scorer-only.
- **egglog does NOT constant-fold f64 literals** unless a rule does it.
- **`Protected*` are distinct functions** from raw ops. Never map a protected name to a raw semantic_id.
- **Bare commutativity/associativity** is fine bounded, fatal at unbounded fixpoint with other expand rules.
- **Determinism**: same input + seed = identical output (tests assert). Sort when choosing among equal-keyed entries
  (HashMap order bit us once).
- **Futhark 0.28.1's "Fuse SOACs" pass can loop for ever**; `futhark_exec` compiles with `--fuse=10` through Futhark's
  own pass list (`Verdict::BoundedFusion`, a compiler workaround, not an e-graph rescue; `docs/RESCUE_futhark.md`).
- **Futhark 0.28.1's prelude is rank-1 and has no `matmul`**; `map2` is curried. Getting either wrong is a type error.
- **Futhark's simplifier folds `x + 0`**: no row may be spelled as an addition of zero.
- **naga rejects some constant expressions** (e.g. `sqrt(-1.0)`): WGSL `sqrt`/`exp` rows are protected.
- **A WGSL `for` becomes naga's `loop_init`, which hangs Apple's dispatch**: the loop row writes a plain `loop`.

## Workflow

- **Commit and push often**: after every change that builds and passes its test, commit your own paths (`git commit
  -- <paths>`; in a shared file only your hunks) and push. Never sit on uncommitted work. Never `git stash`, `git
  reset`, `git rm --cached`, `git add -A`/`.`; no worktrees; never `cargo clean` a shared target. Git identity
  `andrew-gamakon`. Working branch `feat/futhark-kingdom`; main only after the full suites.
- One GPU process at a time, under the GPU lock (`tmp/fixf/gpu_lock.sh`: `take_gpu`, `run_bg`); the cargo lock
  `/private/tmp/claude-501/locks/cargo` for builds only, never held during `cargo test`.
- Keep the live documents (`docs/DESIGN_geneframe.md`, `docs/PLAN_futhark_kingdom.md`) updated in place; no parallel
  new documents. Programs worth keeping are stored as program TEXT, never as genes (no migrations).
- Replies to Andrew: short, plain English, GECCO vocabulary, no jargon; detail goes in commits, tests and docs. Never
  write "nucleotable" (say master symbol table / gene frame).
- Memory: `~/.claude/projects/-Users-andrewmorgan-Dev-gamakon-fuller/memory/MEMORY.md`.
