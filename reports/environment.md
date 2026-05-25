# gamakAST environment & egglog 2.0 API notes

Recorded during Phase 1.0 calibration (2026-05-25).

## Toolchain (this machine)

| Component | Version |
|---|---|
| Platform | macOS arm64 (Darwin 23.4.0) |
| Rust / Cargo | 1.94.1 |
| Python | 3.12.3 (anaconda) |
| maturin | 1.8.7 |
| egglog crate | **2.0.0** (pinned `=2.0.0` in Cargo.toml; locked in Cargo.lock) |
| PyO3 | 0.22 (feature-gated, not yet exercised with code) |

egglog 2.0.0 resolves against Rust 1.94.1 (173 transitive deps) and compiles
in ~12s cold, ~0.2s incremental.

## egglog 2.0 API — what BRIEF.md assumed vs. reality

BRIEF.md was written against a pre-2.0 egglog. The high-level interface
**survived** the 2.0 bump; the capability is in fact richer than the brief
hoped. No upstream blocker — calibration is a GO.

### Crate is now modular
2.0 is split into `egglog`, `egglog-ast`, `egglog-bridge`,
`egglog-core-relations`, `egglog-reports`, `egraph-serialize`. We depend only
on the top-level `egglog` re-export.

### Driving the e-graph
- `EGraph::default()` then `EGraph::parse_and_run_program(None, src)` — the
  stable textual entry point. Used in `src/calibration.rs`.
- Programmatic builders exist too (`egglog::prelude::*`: `datatype!`, `rule!`,
  `rust_rule`, `query!`, `expr!`, `sort!`, plus `exprs::{var,int,float,call}`).
  The `var!`/`span!` macros need a `Span`/`RustSpan` in scope; the
  `exprs::var(..)` helper avoids that and is what we use.
- `EGraph::eval_expr(&Expr) -> (ArcSort, Value)` recovers an e-class root for a
  bound `(let ..)` name.

### Extraction (the load-bearing capability)
Present and first-class in `egglog::extract`:
- `Extractor::compute_costs_from_rootsorts(rootsorts, egraph, cost_model)` —
  costs computed once at init; rebuild the extractor if the e-graph changes.
- `extract_best` / `extract_best_with_sort` — single lowest-cost form.
- **`extract_variants(egraph, termdag, value, nvariants) -> Vec<(Cost, TermId)>`**
  — the K cheapest equivalent forms, sorted, with costs. This is exactly the
  mechanism Phase 1.4 (Rule 6, data-aware extraction) needs.
  - Caveat: multi-variant extraction works only on **eq-sorts** (e-class
    datatypes). For container/primitive sorts it logs a warning and falls back
    to a single variant. Fine for expression-tree denoise; relevant later when
    multi-typed GEP introduces typed sorts.
- `CostModel` is pluggable (`TreeAdditiveCostModel` is the built-in structural
  cost) — this is the custom-cost hook BRIEF.md worried might be absent.
- Convenience wrappers: `EGraph::extract_value`,
  `extract_value_with_cost_model`, `extract_value_to_string`. Calibration uses
  `extract_value_to_string` for the round-trip.

### Saturation
`(run-schedule (saturate (run <ruleset>)))` runs a ruleset to fixpoint.
Per-call wall/node budgets (BRIEF.md hard cap: 1s / 10k nodes) are not yet
wired; that belongs to Phase 1.4 when real rules can diverge.

## Calibration result

`src/calibration.rs` + `examples/00_calibration.rs`: a 5-rule boolean-algebra
ruleset (identity, double-negation, De Morgan ×2, absorption — no bare
commutativity/associativity), 20 hand-written round-trip cases. All 20 pass
under `RUSTFLAGS="-D warnings" cargo test`. `cargo clippy --all-targets` is
clean. The Rust -> egglog -> saturate -> extract -> Rust loop works on this
machine.

PyO3 FFI smoke (BRIEF.md step 4) is deferred to Phase 1.5 — the `python`
feature compiles clean but exposes no functions yet.
