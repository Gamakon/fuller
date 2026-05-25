# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Specification only — no code yet.** The repo contains `BRIEF.md` (full Phase 1 spec), `README.md`, and `.gitignore`. `BRIEF.md` is the source of truth for everything below; read it before writing any code. The directory tree, API signatures, rule definitions, phase ordering, and acceptance criteria in `BRIEF.md` are the contract — follow them rather than re-deriving them.

## What this is

A Rust crate exposed via PyO3 that uses [egglog](https://github.com/egraphs-good/egglog) as a substrate for symbolic expression rewriting. Phase 1 ships exactly one feature: a `denoise` mutation operator that takes a GEP karva chromosome, projects it into an e-graph, rewrites away noise, and projects it back as a smaller equivalent karva chromosome.

The consumer is the sibling project `/Users/andrewmorgan/Dev/kaito/hff/`. The two are **siblings, not parent/child** — never modify `hff/`. `PsetSpec` (pure data, no `geppy` import) is the only boundary between them.

## Build & test (once code exists)

```
pip install -e .          # builds Rust ext via maturin + installs python/gamakAST shim
cargo test                # Rust unit/integration tests (tests/*.rs)
cargo test test_roundtrip # single Rust test target
python -m pytest tests/test_pyo3.py   # Python-side smoke
```

`pip install -e .` must leave a working `from gamakAST import denoise, karva_to_terms, terms_to_karva` in the target env. Requires a Rust toolchain (`rustup install stable`), maturin backend, PyO3 0.22+, Rust edition 2021. Target platform: macOS arm64, Python 3.12.

## The skateboard (Phase 1 dataflow)

```
karva chromosome → egglog terms → saturate(denoise rules)
                                        ↓
                          extract_smallest_with_data_parity
                                        ↓
                          egglog term → karva chromosome
```

Rules 1–5 are pure algebra and fire inside saturation (`src/ruleset/identities.rs`). Rule 6 ("drop data-irrelevant terms") is **not** a saturation rule — it is a post-saturation extraction loop (`src/extract.rs` + `src/eval.rs`): extract the smallest K candidates by structural cost, evaluate each on training data, return the smallest whose R² loss < `tolerance`. egglog cost functions are purely structural; all data-awareness lives in the extraction harness.

## GEP karva invariants the converter must enforce

- A gene = `head` (length `h`, functions or terminals) + `tail` (length `t = h * (max_arity - 1) + 1`, **terminals only — never functions**, a hard GEP invariant).
- The "live" region is found by walking the head left-to-right (BFS), consuming child slots as functions appear. Tokens past the live region are neutral and carried along but don't decode.
- After denoise emits a shorter head, the tail is re-padded with random terminals to satisfy the `t = h*(max_arity-1)+1` rule. Padding uses `random.Random(rng_seed)` over `pset.variables + rnc_values`, and must be deterministic given `(new_head, rng_seed, pset)`.
- `denoise` **never raises** on un-encodable expressions — it returns the original tokens unchanged.
- `semantic_id` (not the geppy function name) is what gamakAST rewrites on. Same operator can have different geppy names across psets; the consumer maps name → `semantic_id` when building `PsetSpec`.

## Hard constraints (from BRIEF.md — do not violate)

- **No sympy. Anywhere** — not in src, tests, examples, or docs. This crate exists specifically to replace sympy. Compute expected test values by hand or in pure Rust/numpy. (`numpy` is allowed in examples for evaluation only.)
- **No `geppy` dependency** — `PsetSpec` is the boundary.
- **No bare commutativity/associativity rewrite rules** — egglog handles these via e-class merging; encoding them as rewrites causes blowup. This is the most common beginner mistake.
- **RHS pattern variables must all appear on the LHS**, or saturation diverges.
- **Conditional rewrites need explicit guards** (e.g. `x/x → 1` needs `when (!= x 0)`).
- **Saturation budget per call: 1s wall clock, 10,000 e-graph nodes**, hard cap. A rule that exceeds it is rejected and logged.
- **Determinism**: same input + same `rng_seed` ⇒ bit-identical output. Tests assert this.
- Pin the egglog crate version; record actual versions in `reports/environment.md`.

## If the egglog Rust API has drifted from BRIEF.md

Document the real API in `reports/environment.md`, adapt the implementation but keep the public Python signatures unchanged. If a critical feature is missing (e.g. extraction by external cost function), **STOP and report — do not invent a workaround.** Same for calibration (Phase 1.0) failure.

## Workflow

- **Do not push.** Leave commits local for the user to review.
- One commit per phase (1.0, 1.1, …), conventional-commits style. Finish a phase and commit before starting the next — do not interleave phases.
- Proceed silently through the work; the deliverable is `reports/phase1_report.md` plus artefacts, not prose progress updates.
- `git status` should be clean when Phase 1 ships.
- Phase 2 (porting SymPy simplification rules into egglog) is a separate later brief. Keep the `src/ruleset/mod.rs` registry extensible so new rule modules can be added without touching `src/lib.rs` or the public API.

## Patterns to read in `hff/` (read-only — do not import)

`BRIEF.md` lists files in `hff/notebooks/` worth reading for the GEP decomposition and karva-serialisation algorithms (`_gene_decompose.py`, `_sympy_to_karva.py`) and as counter-examples (`hff_sr_engine.py`'s `_extract_best`). Read for patterns; build clean.
