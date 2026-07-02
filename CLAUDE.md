# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

fuller is a Rust crate (PyO3-exposed) using **egglog 2.0** as a substrate for symbolic expression rewriting. It replaces sympy for the sibling SR engine in `/Users/andrewmorgan/Dev/kaito/hff/`: deterministic, real-domain, bounded. `hff/` is a **separate project** — only commit there with the user's sign-off, on its own branch.

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
- `src/parity.rs` + `src/bin/parity.rs` — SymPy-parity scorer, **per-family** (`Family::Algebra|Rational|Trig`).
- `src/python.rs` — PyO3: `denoise`, `denoise_karva`, `physics_mutate`, `physics_mutate_karva`, `master_pset`.
- `parity/` — `gen_corpus.py` (offline sympy→Math corpus), `label_corpus.py` (offline family-labeler for the classifier), `corpus/*.jsonl`.
- `nucleotable/` — subsumed design source of truth (referenced by `geneframe.rs`). `stale/` — delivered briefs, history only.

## Non-obvious things that will bite you

- **Rule families are NON-CONFLUENT.** distribute + trig (or + rational) co-saturated explode the e-graph (verified: pegs CPU, killed runs). The scorer keeps them in separate `Family`s; `denoise` uses only the bounded algebra+powers subset. Do NOT merge all rulesets into one saturation. **Always kill-guard a saturation/parity run** so a divergent rule can't peg the machine: `( cmd & PID=$!; for i in $(seq 1 N); do kill -0 $PID 2>/dev/null||break; sleep 1; done; kill -9 $PID 2>/dev/null )`.
- **distribute must NOT go in `denoise`** — it's a normal-form canonicaliser; `extract_variants` over its unbounded equivalent-class hangs. Scorer-only.
- **egglog does NOT constant-fold f64 literals** unless a rule does it (distribute/rational add the folds).
- **`Protected*` are distinct functions** from raw ops (real-domain raw vs the engine's guarded semantics). Never map a protected geppy name to a raw semantic_id — unsound on negatives/zero.
- **The old "no bare commutativity" advice is nuanced** — egglog's own tests ship bare comm/assoc rewrites; they're fine *bounded*, fatal at unbounded fixpoint with other expand rules. Measured finding: commutativity is NOT the parity wall here (≈2/600 pairs); the gaps are structural.
- **Determinism**: same input + rng_seed = identical output (tests assert). HashMap iteration order bit us once — sort when choosing among equal-keyed entries.

## Parity status (the SymPy-replacement metric)

Against frozen SymPy corpora (`parity/corpus/*.jsonl`, generated offline; sympy NEVER in the scoring loop): overall ~33.6% — powsimp 84.9%, trigsimp 47%, radsimp/ratsimp ~15%, simplify 13%. radsimp/ratsimp low % is partly a measurement artifact: sympy rationalises to *larger* canonical forms, not the simpler expression SR wants.

## In flight

- **Simplify-corpus instrumentation** merged in `hff/` (env-gated `GAMAK_SIMPLIFY_CORPUS`) — captures real before→after sympy edits on the SRBench sweep. `parity/label_corpus.py` labels them by family → train a **kingdom classifier** (the learned router that picks which rule family to load, dodging the non-confluence problem). Waiting on the near-miss re-sweep to emit the corpus.

## Workflow & memory

- **Do not push.** Local commits, branch off main, conventional-commit messages.
- The user is impatient — tight status lines, fix-don't-explain, no essays.
- Persistent design context lives in `~/.claude/projects/-Users-andrewmorgan-Dev-kaito-fuller/memory/` — read `MEMORY.md` then `00-design-overview.md`. Key ones: `ownership-contract`, `working-posture-frontier-rnd`, `just-fix-it`, `worktree-agent-cleanup`, `two-part-simplify-and-classifier`.
