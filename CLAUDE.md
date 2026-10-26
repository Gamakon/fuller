# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

fuller is a Rust crate (PyO3-exposed) using **egglog 2.0** as a substrate for symbolic expression rewriting. It replaces sympy for the sibling SR engine in `/Users/andrewmorgan/Dev/gamakon/hff/`: deterministic, real-domain, bounded.

The original Phase-1 spec is **delivered and superseded** — see `stale/BRIEF.md` for history only. Current state is `src/` + `docs/`.

## Build & test

```bash
cargo test                       # Rust tests (plain build, no PyO3 link needed)
cargo clippy --all-targets       # MUST be clean
maturin develop --release        # build + install the Python extension (fuller)
cargo run --example 00_calibration
cargo run --release --bin parity -- parity/corpus/*.jsonl   # SymPy-parity score
```

**Hard rule from the user: zero warnings, none hidden.** Never silence with `_`-prefix or `#[allow(...)]` — fix the root cause. `RUSTFLAGS="-D warnings" cargo test` passing + clippy 100% clean is required before any commit.

## Architecture (what's actually built)

- `src/expr.rs` — `Math` datatype (real-domain ops incl. `Protected*`) + `GUARD_RELATIONS` (is-positive/is-nonzero for guarded rewrites).
- `src/eval.rs` — real-domain evaluator over egglog `Term`s. `sqrt(neg)/log(≤0)/div0 → NaN`; protected ops match the engine's exact semantics (e.g. `ProtectedExp` = uncapped exp, +inf on overflow).
- `src/karva.rs` — karva (GEP chromosome) ↔ `Math` converter, keyed on `semantic_id` (NOT geppy name). Holds `master_pset()`.
- `src/ruleset/` — egglog rulesets: `identities` (algebra), `powers`, `distribute`, `rational`, `trig`.
- `src/extract.rs` — `denoise()`: saturate (algebra+powers only, bounded) → `extract_variants` → score on data → smallest within R² tolerance, else unchanged. The live mutation operator. Never raises.
- `src/physics.rs` — `generate()`: pure one-to-many physics-prior mutation GENERATOR (NO eval/score). Tags candidates `speculative` (caller must extrapolation-gate those).
- `src/snap.rs` — constant snapping (π/e/√2/G… within tol → symbol annotation; Math stays pure-numeric).
- `src/geneframe.rs` — the **nucleotable data model, owned here**: master `SymbolTable`, typed many-hot arity, kingdom = a query. The direction the symbol/pset layer migrates toward.
- **The geneframe holds two row kinds** (design 2026-10-09, `docs/PLAN_population_dag.md`): ordinary chromosomes and homeotic chromosomes (population-wide shared definitions with a stable row id, a root type, memoised row values and a referrer count). An href names a homeotic row by its stable id, never a position in the homeotic tail. Fold is a variation operator: it adds definitions, rewrites ordinary rows to reference them and re-sorts the homeotic tail by use; execution follows a separate level schedule (a definition after everything it references). The memoised top of the homeotic tail is the expressed frame. Determinism is a tested contract: values bit-identical with homeotic rows present or stripped.
- `src/population_dag.rs` — the population DAG on the host (plan task 3): hash-consed nodes, homeotic rows, memos keyed by node, data version, type and input binding, level schedule, nothing evicted. The device arena (variant B in phylu) is not built.
- `src/homeotic.rs` — the earlier per-chromosome homeotic tail (fold of a chromosome's shared subtrees into tail slots, encode back to Karva). Its slot-addressed href is superseded by the stable-row-id href of the population design; see its module header.
- `src/wgsl/` — the WGSL kingdom (feature `wgsl`): naga reader cutting a kernel at its effects into a chromosome over `kingdoms/wgsl` tables, scaffold, interpreter, oracle, mutation, rebuild.
- `src/futhark_egraph.rs` — the Futhark kingdom inside Fuller's e-graph: **one egglog sort per kingdom type** (`FR`/`FA`/`FS`/`FF`), so the sort IS the type and the `acc` distinction is structural. The 38 rows map one-for-one onto constructors, plus two e-graph-only ones (`FId`, `FCompose`) whose candidates are compilation-only. Every rule was checked bit-identical against Futhark 0.28.1. See `docs/EGRAPH_futhark.md`.
- `src/futhark_kingdom.rs` + `kingdoms/futhark_true/` — the 1-D Futhark kingdom: 4 types, 38 rows, "shape is a value, not a type". `src/futhark_shapes.rs` + `kingdoms/futhark_shapes/` — the **shape-classed** kingdom: a shape class IS a type (`mat.n.m` = the interface's `[n][m]`), 11 types, 70 rows. The unification happens ONCE, OFFLINE, in `gen_tables.py`, whose 9 witness tests re-check every shape relation from the emitted text — the table is the trust boundary, because the engine no longer checks shapes. The two kingdoms live in separate `Ty` families (`Ty::Futhark` / `Ty::FutharkShape`). See `docs/ASSESSMENT_first_class_shapes.md` §15.
- `src/parity.rs` + `src/bin/parity.rs` — SymPy-parity scorer, **per-family** (`Family::Algebra|Rational|Trig`).
- `src/python.rs` — PyO3: `denoise`, `denoise_karva`, `physics_mutate`, `physics_mutate_karva`, `master_pset`.
- `parity/` — `gen_corpus.py` (offline sympy→Math corpus), `label_corpus.py` (offline family-labeler for the classifier), `corpus/*.jsonl`.
- `nucleotable/` — subsumed design source of truth (referenced by `geneframe.rs`). `stale/` — delivered briefs, history only.

## The evolution engine moved to phylu

`src/evolve/`, the viewer, `evolve_fit` and the fit runbook now live in
`../phylu` (`docs/SPLIT_fuller_phylu.md`). phylu depends on fuller; fuller must
never depend on phylu. When phylu needs a fuller item, widen it here with a
fuller commit, tested here.

## Non-obvious things that will bite you

- **Rule families are NON-CONFLUENT.** distribute + trig (or + rational) co-saturated explode the e-graph (verified: pegs CPU, killed runs). The scorer keeps them in separate `Family`s; `denoise` uses only the bounded algebra+powers subset. Do NOT merge all rulesets into one saturation. **Always kill-guard a saturation/parity run** so a divergent rule can't peg the machine: `( cmd & PID=$!; for i in $(seq 1 N); do kill -0 $PID 2>/dev/null||break; sleep 1; done; kill -9 $PID 2>/dev/null )`.
- **distribute must NOT go in `denoise`** — it's a normal-form canonicaliser; `extract_variants` over its unbounded equivalent-class hangs. Scorer-only.
- **egglog does NOT constant-fold f64 literals** unless a rule does it (distribute/rational add the folds).
- **`Protected*` are distinct functions** from raw ops (real-domain raw vs the engine's guarded semantics). Never map a protected geppy name to a raw semantic_id — unsound on negatives/zero.
- **The old "no bare commutativity" advice is nuanced** — egglog's own tests ship bare comm/assoc rewrites; they're fine *bounded*, fatal at unbounded fixpoint with other expand rules. Measured finding: commutativity is NOT the parity wall here (≈2/600 pairs); the gaps are structural.
- **Determinism**: same input + rng_seed = identical output (tests assert). HashMap iteration order bit us once — sort when choosing among equal-keyed entries.
- **The Futhark e-graph's `map id` and its fusion rules DIVERGE together.** `map id` puts a class's own `map` node inside that class, and the fusion pattern then matches inside it and builds another one: measured 5 → 7 → 13 → 55 → 1357 e-nodes over five iterations, no convergence after six. Fusion alone is stable. They are in separate rulesets for that reason (`FUTHARK_RULESET` / `FUTHARK_FUSION_RULESET`); do not merge them.
- **Futhark 0.28.1's optimiser does not converge on one legal program** (a reduction over an `acc`, a reversed accumulator-derived operand, and an outer reduction, together). It is a compiler defect, not an illegal gene. Equality-preserving rewrites of the program do NOT rescue it — measured: the `map id` removal, the `map`/`reverse` commutation and the accumulator-flattening all still hang at 180 s. See `docs/EGRAPH_futhark.md`.
- **Futhark 0.28.1's prelude is RANK-1 and has no `matmul`.** `map`/`map2`/`reduce` apply to 1-D arrays of SCALARS; `transpose` is the only 2-D primitive; `zipWith`, `matmul` and `reshape` do not exist. Mapping over a matrix's rows needs an ARRAY-valued function (`map (\row -> …) A`), and `map2` is CURRIED — it needs a TWO-argument function (`[]f32 -> []f32 -> []f32`), which is why the shape kingdom has two function classes. Getting either wrong compiles into a type error, not a wrong answer.

## Parity status (the SymPy-replacement metric)

Against frozen SymPy corpora (`parity/corpus/*.jsonl`, generated offline; sympy NEVER in the scoring loop): overall ~33.6% — powsimp 84.9%, trigsimp 47%, radsimp/ratsimp ~15%, simplify 13%. radsimp/ratsimp low % is partly a measurement artifact: sympy rationalises to *larger* canonical forms, not the simpler expression SR wants.

## In flight

- **The population DAG programme** (`docs/PLAN_population_dag.md`, ahead of the WGSL superoptimiser): tasks 1–3 measured (63 % of subtree occurrences repeated within a generation; 72 % of repeated subtrees survive a generation; the shared DAG does a third of an interpreter's operator evaluations, a quarter with memos). Next: variant B in phylu (Fold in the generation loop, the device arena, checkpoints carrying homeotic rows), then the banded compiler restricted to stable definitions, then the paired A/B/C experiment.
- Earlier item, status not re-checked 2026-10-09: **Simplify-corpus instrumentation** merged in `hff/` (env-gated `GAMAK_SIMPLIFY_CORPUS`) — captures real before→after sympy edits on the SRBench sweep. `parity/label_corpus.py` labels them by family → train a **kingdom classifier** (the learned router that picks which rule family to load, dodging the non-confluence problem). Waiting on the near-miss re-sweep to emit the corpus.

## Workflow & memory

- **Do not push.** Local commits, branch off main, conventional-commit messages.
- The user is impatient — tight status lines, fix-don't-explain, no essays.
- Persistent design context lives in `~/.claude/projects/-Users-andrewmorgan-Dev-kaito-fuller/memory/` — read `MEMORY.md` then `00-design-overview.md`. Key ones: `ownership-contract`, `working-posture-frontier-rnd`, `just-fix-it`, `worktree-agent-cleanup`, `two-part-simplify-and-classifier`.
