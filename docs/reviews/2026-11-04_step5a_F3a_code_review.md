# Code review: plan step 5a, task F3a (`1a7ba021`, `d2fafb00`) and the F2 corrections (`4e730a21`, `2948db62`, `c4d4d90d`)

Reviewer: fuller code reviewer, 2026-11-04. Read against `~/dev/minkymorgan/qdrant/docs/Rules.md`, fuller's
`CLAUDE.md`, `docs/PLAN_futhark_kingdom.md` §5a (the F3a row, 5a.1, 5a.2, 5a.3 and the F3a status paragraphs), the F2
review (`2026-11-04_step5a_F2_code_review.md`) and the P1 review (`2026-11-04_step5a_P1_code_review.md` §5). Every
added and changed line of the five commits was read with `git show`. Nothing was built or run (another agent may be
building); the numbers below are from the commits, the P1 log (`/private/tmp/claude-501/fbench/p1/test_default2.log`
line 182 and 237) and the source.

**Verdict: pass with corrections.** The seam is correct and is library code: no generation loop, no population, no
selection. The bit-for-bit claims against the runner's ledger are really tested. The F2 corrections are done. Two
corrections must land before phylu runs regime R on any problem (R2b), and one before F3b:

1. The size guard's bound is relative to the reference program and kills phylu's regime R start (finding 1). Replace
   it with an absolute bound, measured (finding 2).
2. The guard expresses and prints the whole oversized tree before it checks the size (finding 3).
3. The GPU script was skipped, but a script over the seam is possible now (finding 5).

## Findings

### 1. The size bound of 64 times the reference makes most of phylu's regime-R histogram start score worst. Confirmed.

`src/futhark_evaluator.rs:1660` (`SIZE_BOUND_FACTOR = 64`), `:1761` (`size_bound = reference nodes x 64`), `:2117`.

For histogram the reference has 7 nodes, so the bound is 448. P1 measured phylu's cell-by-cell rows (24 rows,
`test_default2.log:182`): mean 2,420.9 nodes, largest 3,985, largest program 66,863 bytes. All 24 rows passed
`futhark check`, and the 24 rows took 12.5 s in total. The agent's "mostly over" is a floor, not a guess. If k rows are
at most 448 nodes and the rest at most 3,985, the mean of 2,420.9 requires 448k + 3,985(24 − k) ≥ 58,102, so k ≤ 10.
**At least 14 of 24 rows (58 %) are over the bound, and in practice nearly all of them**, because a uniform draw has
no small mode.

Failure scenario: R2b (histogram, regime R, on phylu). Generation 0 has at least 14 rows tied at π/2, and each is
charged to the 2,000-evaluation budget. The pump's refill in R is the same random draw (5a.2, F5 row), so each pump
adds 13 more such rows. A mutant of an ordinary row reads uniform background cells and grows past 448 as well
(histogram's 1-2-cell mutants already reach 307 on `random_gene`'s smaller background). Selection sees a flat
plateau. The run does not simply fail; it measures a different search than the runner's (`futhark_evolve` regime R
draws `futhark_sig::random_gene`, max 204 nodes, all under the bound). So R2b would neither learn nor reproduce the
runner.

The bound also protects nothing that needs protecting here. Histogram's whole cell-by-cell population compiles; the
danger the P1 review found is absolute size (k-means, 1.5 million nodes, 65 MB, more than 120 s and 2.8 GB per
`futhark check`), not size relative to a 7-node reference. The ratio is also unequal across problems: it allows
16,384 nodes on k-means (about 360 KB of Futhark) but only 448 on histogram (about 8 KB).

The root cause is phylu's draw, not the candidates. phylu draws every head cell uniformly from its alphabet, where
functions outnumber terminals. `futhark_sig::random_gene` (`src/futhark_sig.rs:2105-2130`) draws head cells half from
functions and half from terminals. The P1 review (§5, last paragraph) asked for "a change to the kingdom, not a filter
on candidates". The guard is still right to have as the machine's safety net, but the start draw must also be fixed.
Fix: (a) phylu's start, refill and pump draws for an external Futhark kingdom use the same head rule as
`futhark_sig::random_gene` (a P2a/P4 item; the plan should state it as a requirement of R2b, not as "a matter for
P2a/P4"); (b) the bound becomes absolute (finding 2).

### 2. The proposed bound: one absolute node count, 4,096, measured before it is fixed

Evidence for the scale, from the measurements already on record:

| measurement | value | source |
|---|---|---|
| bytes a node, k-means largest row | 65,114,508 / 2,967,385 = 21.9 B | P1 log :237 |
| bytes a node, histogram largest program | 66,863 B, at most 3,985 nodes: at least 16.8 B | P1 log :182 |
| `futhark check`, one 5.8 MB k-means row | 101 s, 2.8 GB: 17 s and 0.48 GB a MB | P1 review §5 |
| histogram cell-by-cell rows, largest 3,985 nodes | every row passes `futhark check`; 24 rows in 12.5 s | P1 log :182 |
| compile limit floor | 30 s a batch; the largest Futhark stage measured is 2.39 s | `src/futhark_batch.rs:220-228` |
| largest mutant or `random_gene` row measured, all problems | 392 nodes (k-means, 1-4 cells) | `d2fafb00`'s table |

A 24-row module that stays near the 30 s floor at 17 s a MB holds about 1.7 MB, which is 72 KB a row, or about
3,300-4,300 nodes at 17-22 B a node. **Proposal: `SIZE_BOUND_NODES = 4,096`, the same for every problem, with the
ratio removed.** It keeps every measured mutant and `random_gene` row (max 392, 10× margin) and histogram's whole
cell-by-cell population (max 3,985). It is 16× k-means's reference (256). Using the single 5.8 MB point and assuming
linear scaling, its worst case is a 24-row module of about 2.2 MB, roughly 38 s and 1 GB. Compare that with what P1
found the guard must prevent: a 0.8 GB module, a 24-minute halving and more than 128 GB.

Before the constant lands, the agent must measure (CPU, no GPU needed), because the scaling above rests on one point:
- 2,000 cell-by-cell draws (`uniform_gene` of the test, which is phylu's `init_row` rule) for each of the six
  problems: the node distribution and the fraction over 4,096. Pathfinder, reduce, scan and a7 are unmeasured.
- One 24-row histogram module of cell-by-cell rows, compiled on the C backend: compile seconds and peak RSS
  (`/usr/bin/time -l`).
- k-means rows cut to about 1k, 4k and 16k nodes: `futhark check` seconds and RSS, to confirm the linear scaling.

If the 24-row module at the bound exceeds the 30 s floor, add a byte budget per module (split the batch before the
compile, the way the binding limit already halves it) rather than lowering the per-program bound below what the
population needs. The bound's doc comment should carry the table, as `d2fafb00`'s does now.

### 3. The guard expresses the whole tree and prints its text before it checks the size

`src/futhark_evaluator.rs:2114-2118`. `self.gen.tree(g)` builds the full `Tree`, `sexp(...)` prints the full
s-expression, and both happen before `nodes > self.size_bound` is tested. The key `oversized <hash of the text>` then
needs the printed text. On histogram this costs nothing. On k-means (F3b) a 3-million-node row is a full tree and a
tens-of-MB string on the host, about the 10 s a row P1 measured for rendering. Only the Futhark render and the compile
are saved. Fix: count the expressed nodes with a walk that stops at bound + 1 (the expression already walks from the
head cells), and key an oversized row by a hash of its coding cells' values (`Genetics::coding_cells`), which decide
its tree. The text then never needs to be printed. Needed before F3b.

### 4. The run file is not checked inside a halving

`src/futhark_evaluator.rs:2181-2187` checks the run file once before `Judge::prefetch`, and `Host::run`'s halving
(`src/futhark_batch.rs:1229-1294`, `split` at 1384) then compiles up to 2n − 1 times without checking it. The P1
review counted about 47 compiles of 30 s, or 24 minutes, for a stuck 24-program batch. Plan 5a.1 ("the run file":
"checks the run file between batches") is met only when a call is one batch. With the size guard fixed this is rarer,
but `killme.trigger` must stop a run within one compile. Fix: pass the run file's path to `Host::run` and check it
before every compile of a half; when it is gone, return the outcomes so far, and the evaluator returns those rows with
`stopped`. Test with a mock host that fails the first compile (as `futhark_batch::tests` mocks a compile).

### 5. The GPU script `f3a_rust_generation.sh` can be written now, and the deferral is not acceptable

The status paragraph says the script "needs a driver over the seam: phylu's P4 or `futhark_measure`". It does not.
The F3a row asks for 3 generations of 24 histogram programs through the seam on `--eval rust`, with one module a call
and the runner's inputs. Fixed gene sets need no loop and no selection, so a GPU-gated test in fuller does it, as
`futhark_wgpu_host`'s `#[cfg(feature = "gpu")]` tests already do. This is a replay, not evolution, and respects
"fuller does not evolve code". `--eval rust` requires the GPU (`src/futhark_evaluator.rs:1737`), so **no test
exercises the Rust path through the seam today**. The script should:
1. run `futhark_evolve --problem histogram --regime R --backend gpu --eval rust --evaluations 72 --seed 1` into
   `OUT_RUNNER`, which writes its ledger;
2. run a `gpu`-feature test that reads that ledger (path from an environment variable), rebuilds each generation's
   genes the way the CPU test does (`generation_zero` and `gene_of`), calls `evaluate_generation` three times, and
   asserts `cost.modules == 1` a call and quality inputs and behaviours bit for bit (speed inputs are wall-clock and
   are excluded, as F2 §5 established).

Write it now; the lead runs it on the GPU queue.

### 6. The bit-for-bit claims are real (no correction)

`tests.rs:219-256` (single) and `:262-279` (batch) compare the angle by `to_bits()` and the ledger columns as
strings. `ledger_columns` (`src/futhark_evaluator.rs:1632-1642`) writes every input in Rust's shortest round-trip form
(`{x:?}`), so string equality is bit equality. The fixture `histogram_r_seed1_ledger.tsv` names its source
(`futhark_evolve` at `e32b39f1`, F2's plumbing run). Generation 0 is rebuilt from `GeneRng::new(1)`, and each text is
checked against the ledger before scoring (`:176-183`). Generation 1's 8 programs are written onto genes, and the
evaluation series (24, then 30) matches `series.tsv`. One gap: generation 1 runs only on `--eval single`, and the batch
test covers generation 0 alone. Adding generation 1 to the batch test costs one compile.

### 7. The cached-gene charge is correct (no correction)

Within a call, rows are grouped by gene slice (`:2107-2128`). Two different genes with one tree share a `source` and
the second is a `quality_cache` hit in `Judge::evaluate` (`:1128-1131`). `prefetch` de-duplicates by source
(`:583-589`), so the repeated rows in `items` (`:2186`, not de-duplicated) cost nothing. A stage-2 re-score of a
cached program charges no evaluation (`:1128`); timing is not an evaluation, as in the runner. Oversized rows are
charged once by key (`oversized_eval`, `:2214-2226`). Their worst score is `Quality::worst`, the same score a program
that fails to run gets (`:856`). Their stage-2 speed inputs are 1.0, the same as an untimed program's (`:1186`). The
charge for a never-compiled oversized row is defensible: the search produced the individual. With finding 1 left
uncorrected, however, it spends 14 or more of every 24 evaluations of an R start on nothing.

### 8. The stage change inside the call (correct; two paths untested)

`:2131-2146`: the best row is the least angle, the first on a tie; stage 2 re-scores every row, and when the run
file goes before stage 2's batch the call returns the stage-1 rows, `stopped` set, and the stage unchanged. That
matches 5a.2. The test (`:287-338`) pre-fills the timing slots on the C backend, so it never runs stage 2's second
module, and nothing tests the "run file gone before the second batch" path. The F3a row asked for a mock host for the
stage change. Fix: a mock-host test (or the GPU script of finding 5) that removes the run file in `after_batch` after
batch 1 of a call whose stage changes, and asserts that the stage-1 rows come back, `stopped` is set and `stage()`
is still 1.

### 9. Regime P's stage depends on `controls` having been called (minor)

`:2010-2012`: the only way regime P starts in stage 2 is that `controls` was called first. If the engine calls
`evaluate_generation` without it, a P run silently starts in stage 1. Fix: put the regime in `EvaluatorConfig`, or
make `evaluate_generation` an error before `controls` has run (the tests that skip the controls would then call it
with `aa_rounds: 0`).

### 10. The F2 review's corrections are done

| F2 item | commit | checked |
|---|---|---|
| §6 the work cap under `wgsl` | `4e730a21` | the cap is in `futhark_sig`, `wgsl::work_guard:22` re-exports it, and `src/lib.rs:102` declares `futhark_evaluator` with no feature gate |
| note 3, the library prints | `2948db62` | no `println!`/`eprintln!` in `futhark_evaluator.rs`; `Messages` holds lines until the caller sets a sink, and a test covers held-then-sink order (`tests.rs:107-120`) |
| note 2, `controls` twice | `c4d4d90d` | `controls_ran` (`:2005-2008`); a second call is an error that says why |
| warnings | `1a7ba021` | the `type_complexity` warning that `2948db62`/`c4d4d90d` carried is fixed by the `Sink` alias, and no `#[allow]`, `_` binding, `todo!` or `dbg!` was added in any of the five commits. Between `2948db62` and `1a7ba021`, though, the tree was not clippy-clean, against the commit rule. |
| note 1, `pub judge`/`pub gen` | **open** | still `pub` (`:1695`, `:1697`). F2 deferred this to "X1, or when F3a gives the seam". F3a now gives the seam, so make both private, with test-only accessors, before phylu's P4 can depend on them. |

## Rules.md

No stubs: the k-means branch returns a stated error that names F3b, which is not a fake. No test was loosened. Own
paths only (`src/futhark_reader/kinds.rs` untouched). The guard is a symptom filter where the P1 review asked for a
root-cause change ("fix root causes, not symptoms"). Finding 1(a) is that root-cause fix; the guard can stay as the
machine's safety net once its bound is absolute.
