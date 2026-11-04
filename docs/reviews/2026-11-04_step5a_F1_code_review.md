# Code review: plan step 5a, task F1 (`82f6dd3e`, `946af1b1`)

Reviewer: fuller code reviewer, 2026-11-04. Read against `~/dev/minkymorgan/qdrant/docs/Rules.md`, fuller's
`CLAUDE.md` and `docs/PLAN_futhark_kingdom.md` §5a.3, row F1. Every changed line of both commits was read.

**Verdict: pass.** One note on the plan text, no defect in the code.

## 1. The task as the plan states it

| plan requirement | done |
|---|---|
| `Host<'a>` owns the kingdom and the signature | yes: `Host` holds `Arc<FutharkKingdom>`, `Arc<Signature>` (`src/futhark_batch.rs` 301-305) |
| `Interp<'a>` owns them | yes: `Interp` holds the two `Arc`s (`src/futhark_sig.rs` 1289-1293) |
| `Genetics<'a>` owns the kingdom, contract, layout, constants' lengths | yes, plus the ring canvas (`examples/futhark_evolve.rs` 1286-1294) |
| `futhark_sig::load` and `signature` return shared handles | yes, `.map(Arc::new)` on the unchanged bodies |
| never `Box::leak` | none added; the only `Box::leak`s in the tree predate F1 and are outside the evaluator |
| tests unchanged; a new test with dropped originals running a C batch | yes, see 2 |

`Judge` and `Run` lose their lifetimes because they hold the above; `examples/futhark_pool.rs`'s `Ctx` changes its two
field types because it stores `load`'s result. Nothing else is touched: every other edit is `x` to `&x` at a call site
(deref coercion from `&Arc<T>` to `&T`), which does not change what is computed. The remaining lifetime-bearing types
(`Member<'t>`, `Inputs<'b>`, `FoldEgraph<'a>`) are per-call values built and dropped inside one function, never held by
the evaluator, so leaving them is right.

Note on the plan, not the code: the F1 row lists `src/futhark_interpreter.rs` among its files. `Interp` lives in
`src/futhark_sig.rs`; `src/futhark_interpreter.rs` holds no borrowing struct, so it correctly has no change. The row's
file list could name `src/futhark_sig.rs` instead.

## 2. Rules

- Test lines changed (all read): `setup`, `kingdom`, `fable_setup` return `Arc` types; `seed_assignments` and
  `values_fingerprint` take `&Interp`; `assert_eq!(*sig, declared(name))` compares the same `Signature` values as
  before. No assertion, tolerance, input or expected value changed. The agent's "type annotations only" holds.
- New test `a_host_and_an_interpreter_built_from_dropped_handles_run_a_c_batch`: the handles go out of scope at the end
  of the block, `Arc::strong_count == 2` proves the host and the interpreter hold the only references, the C batch is
  compared exactly with each program's own quality program, and the interpreter with the stored answers. It is a real
  test, not a stub. Re-run here: passes (7.3 s).
- Zero warnings: `RUSTFLAGS="-D warnings"` build and `cargo clippy --all-targets` clean; no `#[allow]`, no `_` renames.
- fuller does not evolve code: the change is a type refactor only. Commits touch only F1's paths and the plan; the
  untracked `src/futhark_reader/kinds.rs` was not swept in.

## 3. Defects

None found.

## 4. The ledger check

Excluding the 11 `*_ms` columns is justified. They are wall-clock stage times (render, compile, link, run) and differ
between any two runs of the same binary. The plan's own F2 plumbing test states the same exclusion ("byte-identical
except the milliseconds columns"). The 12 compared columns include `eval`, `batch_size`, `quality_inputs`,
`speed_inputs`, `behaviour` and `all_zero`, which carry everything that selection reads in regime R. Limit: regime R on
the C path never exercises speed fitness, so the check does not cover the timing path. F1 changes no timing logic, so
this does not weaken the result.
