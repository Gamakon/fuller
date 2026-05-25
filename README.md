# gamakAST

Egglog-based bidirectional AST hub for symbolic expression rewriting.
Sibling project to `hff/`.

Status: **specification only.** No code yet. See `BRIEF.md` for the
Phase 1 specification (skateboard: denoise mutation operator end-to-end).

## Install (once built)

```
pip install -e .
```

Builds the Rust extension via maturin and installs the Python shim.
Requires a working Rust toolchain (`rustup install stable`).

## What it is

A Rust crate, exposed via PyO3, that uses [egglog](https://github.com/egraphs-good/egglog)
as a substrate for symbolic expression rewriting. Designed to let GEP
karva chromosomes, sympy expressions, SQL trees, and LLM-emitted strings
all map into the same e-graph for rewriting, deduplication, and extraction.

Phase 1 ships one usable feature: a denoise mutation operator that takes
a GEP chromosome and returns an equivalent but smaller one.

## Why not sympy

Sympy's complex-domain assumptions, native-code signal handling, and
exponential simplification paths cost us a full day of work in `hff/`
on 2026-05-25 trying to make `_extract_best` reliable. egglog is the
correct substrate: deterministic, bounded, declarative, fast,
production-tested (Herbie).

## Status

| Phase | Description | Status |
|---|---|---|
| 1.0 | Calibration (boolean algebra in egglog Rust) | not started |
| 1.1 | karva ↔ egglog terms converter | not started |
| 1.2 | Pure-algebra rules (5 identity rules) | not started |
| 1.3 | Real-valued evaluator | not started |
| 1.4 | Data-aware extraction (R²-guarded) | not started |
| 1.5 | PyO3 + Python wrapper | not started |
| 1.6 | Acceptance & report | not started |
| 2.x | Port SymPy simplification rules | future, separate brief |

See `BRIEF.md` for full spec.
