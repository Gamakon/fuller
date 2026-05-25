# gamakAST

Egglog-based bidirectional AST hub for symbolic expression rewriting.
Sibling project to `hff/`.

Status: **Phase 1 skateboard working.** The denoise mutation operator runs
end-to-end (karva chromosome → egglog → saturate → extract → karva chromosome)
and is callable from Python. **Consumers: see [`docs/USAGE.md`](docs/USAGE.md).**
See `BRIEF.md` for the full Phase 1 spec.

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
| 1.0 | Calibration (boolean algebra in egglog Rust) | ✅ done |
| 1.1 | karva ↔ egglog terms converter | ✅ done |
| 1.2 | Pure-algebra rules (5 identity rules) | ✅ done |
| 1.3 | Real-valued evaluator | ✅ done |
| 1.4 | Data-aware extraction (R²-guarded) | ✅ done |
| 1.5 | PyO3 + Python wrapper | ✅ done |
| 1.6 | Acceptance & report | in progress |
| 2.x | Port SymPy simplification rules (power/log, trig) | in progress (feature branches) |

Work lands on feature branches (calibration, denoise core, converter); not yet
merged to `main`. See `BRIEF.md` for the full spec and `docs/USAGE.md` to use it.
