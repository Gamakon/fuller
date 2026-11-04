# Code review: plan step 5a, task P1 (phylu `c3862062`, fuller `3520f674`)

Reviewer: fuller code reviewer, 2026-11-04. Read against `~/dev/minkymorgan/qdrant/docs/Rules.md`, phylu's and
fuller's `CLAUDE.md`, and `docs/PLAN_futhark_kingdom.md` §5a.1 and the P1 row of §5a.3. Every changed line of both
commits was read (phylu: `Cargo.toml`, `src/evolve/engine.rs`, `src/evolve/futhark_typed.rs` (new),
`src/evolve/mod.rs`, `src/evolve/shape_typed.rs`; fuller: the plan's status paragraph). The test logs
(`/private/tmp/claude-501/fbench/p1/test_default.log`, `test_default2.log`) and both agents' session transcripts
for 01:10-01:38Z were read for sections 5 and 6. Nothing was built or run by the reviewer.

**Verdict: pass with corrections.** The engine part is done as planned and reuses the engine's own mechanisms. Two
corrections: the plan's test covers two of the six problems, not six (4.1), and a stale dev-dependency (4.2). The
measurement in section 5 is correct, and it is the most important result of the task. It shows that a random k-means
individual cannot be compiled, and neither G1 nor the compile time limit protects against that.

## 1. The task as planned

| P1 requirement | done |
|---|---|
| `evolve::futhark_typed` builds the shape, sampling, walk-only stage and paged device parts from the evaluator's contract and layout | yes. `FutharkTyped::new` (futhark_typed.rs 57-68), `device_parts` behind `gpu` (88-101) |
| `shape_typed`'s `ShapeRun::new` and `check_layout` moved, not copied | yes. The removed lines of `shape_typed.rs` and the new ones in `futhark_typed.rs` are the same statements, reformatted. `shape_typed` now calls `FutharkTyped::new` and `device_parts`. There is no second copy |
| A7 unchanged | yes, by reading: the same root, spec, shape, layout check, function and terminal lists, sampling, walk stage, page limit, `DeviceParts` (empty pools, `int_span` 0) and `with_parts_paged` call. The only addition is that the contract, layout and constant lengths are cloned into `FutharkTyped`, which costs a little memory and changes nothing. `shape_typed` is gpu-only, so A7 compiled (clippy `gpu` clean) but did not run |
| `Engine::new_with_table` takes `kingdom = "futhark"` with a placeholder `Data` | yes. `Given::Futhark`, `Data::placeholder`, the half-configured check, `futhark_config_errors` (engine.rs 3906), the carrier layout, and the typed device arm (4581) built like the TSR arm |
| every test of `EXTERNAL_KINGDOM` keyed on the kind, not the name | yes. `EXTERNAL_KINGDOM` was used only by `combinations_for`. It and `Engine::combinations` now take `KingdomKind`, which is decided once at construction (`External` when the engine was given Futhark parts or an external table). `grep` finds no other name-keyed external test. The regex check (`is_regex_kingdom`) is still keyed on the name, which is outside P1's scope |
| tests: `check_layout` for **all six problems** at their program sizes, host only, k-means timed | **partly**. Only histogram and k-means (4.1) |

## 2. Duplicated evolution machinery

None added. `futhark_typed` holds only the kingdom's device parts. The engine builds its device with the existing
`TypedDevice::with_parts_paged` and `GpuEvaluator`, the same way as the TSR arm, and `shape_typed` now takes its parts
from the same function. The one repeated piece is small: `layout_carrier` (futhark_typed.rs 78-82) is the third copy
of the same four-line formula (`regex_typed.rs` 102, `tsr_typed.rs` 67). It should become one helper the next time one
of the three is touched. `shape_typed`'s one-island loop still runs beside the engine, but it was there before P1, and
plan 5a.5 removes it.

## 3. Rules.md and CLAUDE.md compliance

- No stubs, no `#[allow]`, no `_`-prefixed names. Clippy was clean with default features and with `gpu`
  (transcript, 01:25:00-01:27:41Z). 252 tests passed (`test_default2.log`: 222 + 27 + 2 + 1). The GPU test
  `the_engine_builds_on_the_futhark_kingdom` was built but not run, as the brief required.
- **A check was dropped after it failed.** The first run (`test_default.log`) ran `futhark check` on every k-means row,
  and row 1 got no answer within 120 s. The committed test runs no compiler check for k-means. The commit message and
  the test's doc comment say so, and the plan never required a compiler check, so this does not loosen a planned
  test. But the failure is a result, not a test problem (section 5).
- `let _ = std::fs::remove_dir_all(&out);` (futhark_typed.rs 259, 316) throws away the error and deletes a whole
  directory tree. It is the test's own temporary directory, so this is minor. A cleanup that fails should say so.
- Own paths: the phylu commit holds only P1's five files. The phylu `Cargo.lock` was changed by the build (fuller's new
  `rand` 0.9, `rand_chacha` 0.9 and `statrs` 0.18) but was left out of the commit. So `c3862062` cannot build with
  `--locked` (4.3).

## 4. Defects

1. **The plan's test covers two of six problems.** The P1 row asks for `check_layout` on all six problems at their
   program sizes. Only `a_histogram_…` (futhark_typed.rs 263) and `a_kmeans_…` (272) exist. Reduce, scan, pathfinder
   and a7 have no test. G1 depends on all six layouts. Add the four, each a few seconds (they are N = 32 layouts).
2. **Stale dev-dependency** (phylu `Cargo.toml` 51-53): `fuller = { path = "../fuller", features = ["wgsl"] }`, with
   a comment saying it stays "until fuller's F3a moves the work cap out of it". That move landed at fuller `4e730a21`
   (01:10:37Z, the F2 review fix, not F3a), 25 minutes before P1 was committed, and `futhark_evaluator` is ungated at
   `3520f674`. Remove the line. It turns on `wgsl` (naga, toml) in every phylu test build for no reason.
3. **The tests ran against an uncommitted fuller tree.** phylu reaches fuller by path, and at 01:27:54Z fuller's
   working tree held F3a's uncommitted `futhark_evaluator.rs` changes (still uncommitted now). So the "252 pass" is
   not tied to any fuller commit. Together with the uncommitted `Cargo.lock`, the result cannot be reproduced from
   the two commits named. When F3a lands, rerun phylu's tests at that fuller commit and record both hashes.
4. **The empty symbol table** (engine.rs, the `FUTHARK` arm of the table match). The engine reads `self.table` in 38
   places, for example `gene_codec` (6831) and `beam` (7446). For Futhark only `futhark_config_errors` keeps them
   from running, because it names `snap_every`, `fold_every`, `beam_every` and the others. That is enough for P1. P2a
   adds pump and fold beats for an external kingdom, and each of those paths must go through `futhark_typed` and not
   the table, or it will fail on an empty contract.
5. **Minor:** no test covers `KingdomKind::External` for `Given::ExternalTable` (the trivial combination). No caller
   passes a table today.

## 5. The measurement: random k-means individuals express to 1.5 million nodes

**Verified.** `test_default2.log` line 237 has: kmeans, program size 192, 14,359 types, gene 27,563,026 words; mean
1,526,284.5 nodes, largest 2,967,385; largest program 65,114,508 bytes; evaluator 82.6 s, parts 123.0 s, 24 rows
231.7 s. Histogram, for comparison: mean 2,421 nodes, largest program 66,863 bytes, every row passes `futhark check`.
The count is the expressed tree (`count(tree)` after `express`), and the walk agrees with it on every row. The 5.8 MB
timing (101 s, 2.8 GB for `futhark check`) is from the agent's `/usr/bin/time -l` on row 0. In the first run, row 1
(65 MB) did not finish `futhark check` in 120 s.

**What it means.**

- **A random start (regime R) cannot work for k-means.** One row is about 30 MB of Futhark on average. A 24-row batch
  module is about 0.8 GB of source, and type checking alone takes more than 100 s and gigabytes of memory per row.
  Rendering takes about 10 s a row on the host. A k-means run from random genes would spend its budget on compile
  timeouts. The same applies to every random row a refill or the R pump draws.
- **Regime P starts small but is not safe either.** The reference program is written over a random gene
  (`futhark_search.rs` 1059-1060, `futhark_evolve.rs` 304), so every cell the reference does not use holds random
  background. A point mutation that turns a terminal into a function, or a device crossover over the whole row, reads
  that background and can express a subtree as large as the random case. How large P's mutants and children get is
  not measured. Measure it (the node and byte distribution of P mutants at 1-2 expressed cells, and of crossover
  children) before P2a's k-means test and before P3 sets the rates.

**Does G1 cover it? No.** G1 measures the device: the genome store (fixed width, 110.4 MB a row whatever it expresses)
and the walk. It never renders or compiles. Its expected column also leaves out the walk output buffers: 32 B (8 words)
times the shape's node bound for every row, paged at `binding_limit()`. G1 should state that number before it runs.

**Does the compile time limit cover it? Only partly.** `max(30 s, 30 x median)` (`futhark_batch.rs` 227-244) is
measured from the compiles that finished:
- When nothing finishes, the 30 s floor holds. But halving a 24-program batch down to single programs is about 47
  compiles of 30 s each, roughly 24 minutes a generation that produce nothing. Each killed batch's source (about
  0.8 GB at the top level) is also saved under `bugs/`.
- When some huge programs do finish slowly, the median goes up and the limit goes up with it (a 100 s median gives a
  50-minute limit). The rule follows the population; it does not cap it.
- Nothing limits memory. `futhark check` used 2.8 GB on 5.8 MB of source. Scaled linearly, a 24-row module of 30 MB
  programs would exceed the machine's 128 GB, and memory exhaustion like that has crashed this machine before.
- Rendering on the host (about 10 s a row) is not limited at all.

The plan needs a decision here before any k-means run. Two options: cap the expressed size the layout allows at
N = 192 and 14,359 types, or fill the background of P genes so that mutation and crossover cannot express it at
random. Either one is a change to the kingdom, not a filter on candidates.

## 6. The cargo lock the agent removed

What happened (from the transcripts): at 01:23:19Z P1 ran `mkdir locks/cargo && cd phylu && clippy …; clippy …;
test --no-run …; rmdir locks/cargo`. The `mkdir` failed because F3a held the lock for its clippy loop. The `&&` only
skipped the `cd`, so the builds ran without the lock, and the final `rmdir` (about 01:24:40Z) removed F3a's lock. P1
recreated it at 01:24:45Z.

Was F3a affected? **No result of F3a's was changed.**
- phylu builds into `phylu/target` and fuller into `fuller/target`, so there is no shared build directory and no shared
  cargo lock. The `locks/cargo` directory is our own one-build-at-a-time convention, not cargo's.
- F3a's clippy loop (default, `wgsl`, `gpu,wgsl`) finished clean at 01:24:56Z. Its `rmdir` then removed the lock P1
  had recreated, which matches P1 taking the lock at 01:25:00Z. F3a's full test run (01:25:03-01:37:17Z) passed
  783 of 783.
- What did go wrong: for about 80 s two builds ran at once (P1's clippy and F3a's), against the one-build rule. P1's
  unlocked clippy also built F3a's half-edited fuller and failed on F3a's in-progress warnings (`futhark_evaluator.rs`
  62, `tests.rs` 299). That result was thrown away, and the clean rerun came at 01:25-01:27Z. P1's 437 s test run and
  F3a's 731 s suite then ran at the same time, both compiling Futhark. That is allowed (the rule is about builds), and
  both passed.
- For the briefs: `mkdir lock && a; b; rmdir lock` runs `b` and the `rmdir` even when `mkdir` fails. A brief should
  give `until mkdir …; do sleep 5; done; { …; }; rmdir …`, which is the form P1 used on its retry.
