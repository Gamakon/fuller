# Advisor's review of step 5b, "The Futhark evaluator moves to phylu" (plan commit 20086ec9)

Reviewed 2026-11-04 against Andrew's ruling of the same day: fuller holds symbolic programs (kingdoms, the tree,
gene encoding and decoding, rendering, the e-graph, snapping, folding); phylu holds evolution and every fitness
measurement (the schemes, compiling and running, the GPU kernels, scoring on data), the runners and the checkpoints;
phylu gets a general way to call external compilers, simulators and programs for fitness, as DEAP-like frameworks
do; the repositories merge after this work. Read: `docs/PLAN_futhark_kingdom.md` 5a (context, 5a.1, 5a.2, 5a.3) and
5b; phylu `src/evolve/external.rs` and `engine.rs`'s `External`; the import graph of fuller's `src`, `examples` and
`tests`; the three GPU scripts and the F3b log of 2026-11-04 10:29Z.

## Verdict

**Run 5b after five changes.** The file split follows the ruling, and the dependency direction is stated correctly.
But (a) 5b only relocates Futhark code: it gives phylu no general way to call an external program, which is the
second half of the ruling; (b) the move table misses sixteen fuller files that import moved code, so fuller would not
build after M1b; (c) two pieces of the move are wasted work given the coming merge (the three helper crates, and
about 6,200 lines of deprecated runner code (both repositories) moved only to be deleted at X1); (d) the F3b check fails on one
comparison that cannot be stable, and must be fixed before it can serve as the proof.

phylu's engine side is already general: `engine::External` is a set of closures (fitness, fresh, describe, seed,
columns, extras, state, rates) with no Futhark name in its type, and the mock fitness of `p2a_mock.sh` already drives
it. What is missing is the evaluator side: the part every compiler or simulator needs (render a program, skip a
program scored before, run a batch under a time limit, split a failing batch, keep the failing source, check the run
file between batches, turn the vector into the scheme's angle). Today all of it is buried in `futhark_batch::Host`
and `futhark_evaluator::Judge`.

## Recommendations

### 1. Add the general external-evaluator interface (the ruling's second half), small, after the pure move

One phylu file, `src/evolve/program_eval.rs`, about 300 lines, extracted from code that exists (no new behaviour for
Futhark):

```rust
/// Anything that turns program texts into HFF input vectors: a compiler and its runtime, a simulator, a script.
pub trait ProgramTool {
    /// The program an individual expresses. An individual that does not express is a bug of our software.
    fn render(&mut self, genome: &[u32]) -> Result<String, String>;
    /// Score many programs in one go (one compiled module, one simulator launch). A failure of the whole batch is
    /// returned as such; the evaluator then splits the batch in two and tries each half.
    fn score_batch(&mut self, texts: &[&str], limit: std::time::Duration) -> Result<Vec<Vec<f64>>, BatchFailure>;
}

/// The general part: the cache by program text (a repeat costs nothing; the new-evaluation count is the budget),
/// the time limit (a floor and a multiple of the median, as `COMPILE_LIMIT_FLOOR` / `COMPILE_LIMIT_FACTOR` today),
/// the halving, the failing source kept under OUT/bugs/, the run file checked between batches, and the fitness
/// scheme's angle of each vector. `into_external` gives the engine its `External`.
pub struct ProgramEvaluator<T: ProgramTool> { /* tool, cache, scheme, limits, out */ }
impl<T: ProgramTool> ProgramEvaluator<T> { pub fn into_external(self) -> External; }

/// The simplest tool: write each program to a file, run a given command line on them under the limit, read one
/// line of numbers a program. This is "call any external program" with no Rust written for it.
pub struct CommandTool { /* argv, file suffix */ }
```

Process launching reuses fuller's `futhark_exec::output_with_timeout` as it is; it is not copied (the merge puts
both in one workspace). Three users prove it is general: the Futhark evaluator, the P2a mock (a hash of the gene,
rewritten as a `ProgramTool`), and a `CommandTool` test that scores programs with a two-line shell script. Do not
pull the Futhark-only parts into the general layer: the two fitness stages and their re-score, the timing pairs
against the reference program, the resident k-means sample and the champion's test case stay inside the Futhark
tool. If a general piece does not fit Futhark without changing its numbers, leave it in the Futhark tool and note
it; the GPU checks decide.

Plan text to change. After the paragraph "The boundary between the engine and the evaluator (P2a's hook)", add:

> **The general external evaluator (Andrew, 2026-11-04: "a general way to call external compilers, simulators and
> programs").** phylu `src/evolve/program_eval.rs`: the trait `ProgramTool` (render a genome, score a batch of texts
> under a time limit), `ProgramEvaluator` (the cache by text and the budget count, the time limit, the halving, the
> failing source under OUT/bugs/, the run file between batches, the scheme's angle, `into_external`) and
> `CommandTool` (any command line). Extracted from `futhark_batch::Host` and `Judge`, not written new. The Futhark
> evaluator is one `ProgramTool`; the P2a mock is another; a `CommandTool` test with a shell script is the third.

And add a task row after M1g:

> | M1c | phylu | 2 | the general external evaluator above, extracted; the Futhark evaluator and the P2a mock use it | phylu's
> full suite; the `CommandTool` test; the CPU plumbing tests bit for bit | M1g's three GPU scripts again, the same
> counts |

with the DAG line `... ─> M1g ─> M1c ─┬─> F4 ...`.

### 2. Complete the move table: sixteen fuller files import moved code and are not in it

After M1b, nothing left in fuller may name a moved module, or fuller does not build. These do today (found by
searching every `src`, `examples` and `tests` file for the moved module names):

| fuller file | what it uses | where it goes |
|---|---|---|
| `src/futhark_sig/tests.rs`, `tests_a7.rs`, `tests_kmeans.rs`, `tests_kmeans_seed.rs`, `tests_view.rs`, `tests_micro.rs`, `tests_checklist.rs`, `tests_flat.rs`, `tests_pool.rs` | `futhark_batch`, `kmeans_converged`, `futhark_search`, `futhark_rings`, the measuring half (`compile_c`, `run_exe`, `load_cases`) | M1a sorts each TEST, not each file: a test of the kingdom (a rendering compiles, the interpreter agrees with the compiler) stays and calls `futhark_exec`; a test of a measurement moves with it |
| `src/futhark_adf_tests.rs`, `examples/futhark_adf_saturated.rs` | `futhark_search::suite` | `suite` and `input_set` (the deterministic input suite) move into the kingdom half of `futhark_sig` (they are test inputs of the kingdom), or with the moved code if only measurement uses them |
| `examples/futhark_pool_check.rs`, `examples/futhark_problem_cases.rs`, `examples/futhark_pixel_reference.rs`, `examples/kmeans_cases.rs` | the measuring half, `kmeans_converged`, `futhark_wgpu_host`, `futhark_rings` | phylu, with the code they call |
| `tests/runner_standard.rs` (fuller's) | `futhark_search::{evolve, ...}`, the `futhark_a7_evolve` marker | its Futhark entries move to phylu's `tests/runner_standard.rs` |

Keep in fuller, beside `futhark_exec`, the three small functions that compile one program and run it on one input
(`compile_c`, `run_exe`, `write_binary`/`read_binary`): many kingdom tests use them to check that what fuller renders
is legal Futhark that computes what the interpreter says. That is the kingdom's own check, the same reason the plan
keeps `futhark_exec`; moving them would drag those tests out of fuller.

Plan text to change. In M1b's "what" cell, after "the tests move with the code", add: "every fuller file that
imports a moved module is listed in the move table and placed (the table of the advisor's review, recommendation 2);
`compile_c`, `run_exe` and the binary reader and writer stay in fuller beside `futhark_exec`". In M1a's proof cell,
add: "and `grep` finds no moved module named in any fuller file that stays".

### 3. Do not move the three helper crates now

`futhark-rust-rt/`, `futhark-rust-backend/` (the compiler patches) and `futhark-wgpu-shim/` are separate crates with
their own lock files and build directories (2.5 GB of build output), prebuild scripts, a fetch script and paths
written into GPU scripts and issue files. Moving them forces a cold rebuild of the runtime and a sweep of every path,
and the merge will move them once more, into the workspace. They are build tools, not fuller library code, so
leaving them in place for a few weeks does not break the ruling.

Plan text to change. Replace the table row "`src/futhark_rust.rs`, `futhark-rust-rt/`, `futhark-rust-backend/` (the
compiler fork and its patches)" with "`src/futhark_rust.rs` (the runtime's crates `futhark-rust-rt/`,
`futhark-rust-backend/` and `futhark-wgpu-shim/` stay where they are until the merge; the moved code finds them
through one path constant, set to fuller's checkout, in place of `CARGO_MANIFEST_DIR`)". Drop "the build scripts'
paths (`CARGO_MANIFEST_DIR` of the runtime and the shim) fixed" from M1b and write "one path constant for the
runtime's crates".

### 4. Do not move the deprecated runners only to delete them: freeze them as binaries

`futhark_search.rs` (1,560 lines), `examples/futhark_evolve.rs` (3,448), `futhark_a7_evolve.rs` and phylu's
`shape_typed`/`shape_fit` exist for one reason now: to be the reference that R0 to R4 compare against. A reference
must not change, so its source needs no new home: build each runner once from the commit before M1b (tag it
`pre-5b`), keep the binaries at their own paths (the house rule "own binary path per run"), and run R0 to R4 and the
GPU scripts' first step from them. Then the move deletes those files in place of rewriting their imports, which also
removes the circular question of `futhark_search` (it imports `futhark_gpu`, which moves). The slope check and the
other non-evolution commands are run from the frozen `futhark_evolve` binary until F8 replaces them. A past result
is still reproduced at its own commit, as `docs/FITNESS_SCHEMES.md` says.

`futhark_pixel_evolve` is the only runner of the pixel family; freeze it the same way, and say in the plan that a
pixel runner on phylu is later work.

Plan text to change. Replace the table row beginning "`src/futhark_search.rs` (deprecated ..." with: "the deprecated
runners (`futhark_search.rs`, `examples/futhark_evolve.rs`, `futhark_a7_evolve.rs`, `futhark_pixel_evolve.rs`, phylu
`shape_typed`/`shape_fit`): built once at the tag `pre-5b` into `/private/tmp/claude-501/fbench/frozen/`, then
deleted in M1b, not moved; R0 to R4 and the GPU scripts run the frozen binaries; the slope check runs from the frozen
`futhark_evolve` until F8". X1 shrinks to the documentation and inventory edits, and its gate (Andrew reads R0 to R2b,
R3 and R4 have run) stays the gate for calling the regression done.

### 5. Fix the F3b check before it is used as the proof

The F3b run of 10:29Z failed on one row of generation 0, program
`(tabrows (row.k.d (histrows.add.n (map2.add zeros zeros) points) i_k))`: path, evaluation flag, quality inputs
(every case `1.0, 1.0, 0.5`, the worst) and the all-zero flag were equal; the output fingerprint was `41ccb695eabdef00`
in the runner and `89e0389c95b5192a` in the test. The program's centroids are garbage, and garbage need not be the
same garbage in two processes. The fitness reads only the quality inputs, so the check must too for such a program.

Change `src/futhark_evaluator/tests.rs` (around line 764): compare path, evaluation flag, quality inputs and the
all-zero flag on every row; compare the output fingerprint only when the row's quality inputs are not all at the
worst value; print the rows whose fingerprints differ, with their text. Apply the same rule in the moved copy.

Separately, record a bug, not a check rule: a program whose output differs between two runs on the same input is
reading something not set by the program (or a reduction whose order changes). The determinism contract is ours;
find which (the rust-wgpu fork, or the program reading an empty histogram) and fix it. It does not block 5b.

Plan text to change. In M1g's "proof" cell, add: "F3b compares the output fingerprint only for rows whose quality
inputs are not all at the worst value (fixed in fuller before M1a, its GPU script rerun and passing in fuller first)".

### 6. The smallest safe order

1. **In fuller, before any move:** the F3b comparison fix (recommendation 5); rerun `f3b_kmeans_generation.sh`
   until it passes in fuller; tag `pre-5b`; build and store the frozen runner binaries (recommendation 4). Now the
   three GPU scripts have a passing baseline with recorded counts.
2. **M1a**, in fuller: split `futhark_sig.rs`, and sort every test of recommendation 2 into "stays" or "moves",
   behaviour unchanged. Proof: fuller's full suite, and the search finding no fuller file that names a module to be
   moved, except the files that move.
3. **M1b**, in both repositories, one commit each, landed together: `git mv` the files, rewrite their `crate::`
   imports to `fuller::` paths, define `FreshRequest`/`FreshPurpose`/`FreshIndividual` in phylu, delete the frozen
   runners, set the one path constant for the runtime's crates. No edit to a test but its paths. Proof: both full
   suites, default and GPU features, with `RUSTFLAGS="-D warnings"` and clippy clean; the CPU plumbing tests
   (histogram R and k-means P ledgers) bit for bit in phylu.
4. **M1g:** `f3a_rust_generation.sh`, `f3b_kmeans_generation.sh` (both pointed at phylu's test names and the frozen
   runner) and `p2a_mock.sh`, one at a time on the GPU. Proof: the same tests pass with the same counts as step 1.
5. **M1c** (recommendation 1): the general evaluator extracted. Proof: phylu's full suite, the `CommandTool` test, the
   plumbing tests bit for bit, and the three GPU scripts again with the same counts.
6. Then F4 to F8 and the rest as the plan has them.

Each step is a commit that can be reverted alone, and every move is checked by tests that did not change.

### 7. What to leave for the merge

The merge dissolves the line between the repositories, so do not spend time on it now:

- the second move (M2: `fitness.rs`, `runner.rs`, `telemetry.rs`, `run_file.rs`, the quality parts): leave it to
  the merge; it is a rename inside one workspace then;
- the eight `CLAUDE.md` sentences: fix them once, after the merge, from the workspace's real layout (Andrew's OK
  needed either way); before then, one line at the top of the "Architecture" section saying "evaluators are moving to
  phylu (plan 5b)" is enough;
- the `geneframe.rs` and `population_dag.rs` split (sentence 5): a merge question, not 5b's;
- no new public-interface layer between fuller and phylu beyond `fuller::` paths: anything built to keep the
  repositories apart is thrown away at the merge.

Plan text to change. Replace "What stays for now and moves later (a second move, M2, needs Andrew's OK)" with "What
stays until the merge (the merge makes it a move inside one workspace)", and in the `CLAUDE.md` list's heading add
"(to be rewritten once, after the merge)".

### 8. The ruling, file by file: two small points

- `futhark_rings.rs` and `futhark_operators.rs` move (they are variation operators): correct. Note that
  `telemetry.rs` names `futhark_rings` only in a comment, so it does not block the move.
- `futhark_cost.rs` (a cost read from the program text) and the interpreters staying in fuller is right: they
  describe the program, and phylu's evaluator calls them. `futhark_exec.rs` staying is acceptable as the kingdom's
  own compiler check (recommendation 2); say in its module header that fitness evaluation does not grow there, so no
  second evaluator appears in fuller.

### 9. Dependency direction

The plan states it correctly: phylu depends on fuller, never the reverse. Recommendation 2's search is the test of
it; add it to M1a's and M1b's proof cells. After M1b fuller must build and pass its suite with phylu's checkout
absent (move phylu aside, or build fuller in a clean checkout), which proves no path or build script of fuller
reaches into phylu.
