# Review: step 5a, phylu runs the Futhark problems (advisor, 2026-11-03)

*Of `docs/PLAN_futhark_kingdom.md` §"Step 5a" at fuller `decdca57` (`feat/futhark-kingdom`), against fuller
`CLAUDE.md`, `docs/SPLIT_fuller_phylu.md`, `examples/futhark_evolve.rs`, `src/futhark_search.rs`,
`src/futhark_batch.rs`, `src/futhark_sig/kmeans_converged.rs`, `docs/KMEANS_FITNESS.md`, and phylu `CLAUDE.md`,
`src/evolve/engine.rs` (the generation loop, `External`, `evaluate_external`, `fresh`, `refill`, `pump`,
`new_with_table`), `src/evolve/shape_typed.rs`, `examples/shape_fit.rs`, `src/evolve/checkpoint.rs`,
`src/evolve/typed_device.rs`, `src/evolve/marriage.rs`. Read-only; no code was changed, nothing ran on the GPU.*

## Verdict

The plan puts the right things in the right crate: every piece of selection, the islands, the cohorts and the pump
stay in phylu's engine, and fuller is left with a library object that scores a generation and reports. The seam
(`evaluate_generation` over the whole generation, the HFF input vector and the scheme's angle computed in fuller) is
the right shape for the engine's `External` hook, and the regression is sensible because it compares measures, not
bits. It is not yet buildable as written. The engine's external path has never run (nothing in phylu calls
`set_external`), and it lacks five things the plan leans on; those make P1 and P2 the critical path and larger than
the table says. The plan's "one compiled module a generation" is not what the engine does on a beat generation. The
pump's refill in regime P will be random genes unless F5 says otherwise. F2 and F3 are each days, not hours. The
four open questions are answered below; the k-means population fits the device with room to spare, on measured
limits.

## Findings, most severe first

### 1. The engine's external path is untried and short of what the plan needs (P1, P2)

- Nothing calls it today: `grep -rn set_external phylu/src phylu/examples` finds only `engine.rs`. The comments'
  caller (`wgsl_fit --model-rewrite`) is gone. P2's mock test is therefore not a nicety; it is the first time the
  path runs at all, and P1 is the first time it runs on a typed layout.
- The hook is keyed on a name, not a kind: `EXTERNAL_KINGDOM = "safeWGSL"` and `combinations_for` test
  `config.kingdom == EXTERNAL_KINGDOM` (engine.rs 3861-3870). The plan knows (P1), listed here because every other
  item below sits behind it.
- The refill places the hook's genomes in the FIRST pair's intake only: `Engine::fresh` uses
  `let intake_hi = self.config.pop_intake` and writes row `intake_hi - 1 - k` (engine.rs 5359-5374); `fresh_plan`
  builds one `Population` for every pair's refill (5329-5338). With one pair this is fine; with the two-pair A/B
  the plan proposes later, pair 1 refills with random genes. P2(e) must place per pair.
- The fresh hook carries genomes only: `ExternalFresh = Box<dyn FnMut(u32, usize) -> Vec<Vec<u32>>>` (3876), no
  origin, no parents, no error. The plan's `FreshIndividual` with its mutation event needs a widened hook; P2(e)
  says so, but the DAG puts P2 before F5, so the two must agree on the type first (write the type in F2's module
  and have P2 import it).
- The budget is the engine's, not the evaluator's: `evaluate_external` charges `eval_counts.add(source, 1)` for
  every row the hook scored (4798-4801), cache hit or not, and `max_evaluations` reads `eval_counts.total()`
  (8896). The Futhark budget is programs not scored before, by text (`Judge::evaluate`, futhark_evolve.rs
  1218-1227). P2(a) must let the hook return what it charged.
- The stage change can be done through the hook as it stands: the hook receives `&mut Generation` and the engine
  reads EVERY row's fitness back afterwards (4793-4804), elites included, so re-scoring all rows in stage 2 works
  without an engine change. What needs the engine is the hall-of-fame restart and the `stage_change` event (P2 b):
  `remember` keeps the best of the fit by angle (8968), and stage-1 angles are not comparable with stage-2 ones.
- `Checkpoint` has no slot for a hook's state (checkpoint.rs 60-116); the hall-of-fame file's columns are SR's
  (`log10_p mse_train r2_train`, engine.rs 8871-8875). P2(d) and P4 are right to own these; they are not small.
- `External.seed` fills rows 0.. of the whole population (8661-8668): with intake 16 and champion 8 the reference
  program and its first 15 variants land in the intake and the next 8 variants in the champion island at
  generation 0. The deprecated runner had no islands, so this is new behaviour of regime P; state it in P4's
  card and in the regression table (the champion island is not empty at the start).

### 2. "One compiled module a generation" is not what the engine does on a beat generation (F3, 5a.1)

The engine calls the fitness after variation (`evaluate_fresh`, 8957) and AGAIN after every beat that lands rows:
the pump (9266), the fold (9169), the snap (9151), the beam (9000). On a pump generation the hook is called twice,
so the evaluator compiles two modules: the children's, then the refill's. For k-means the champion's test case is a
third (`converged_on_test` builds its own module, futhark_evolve.rs 1073-1085). This is the engine's order and
should not be changed for this kingdom; the plan should say "one module an `evaluate_generation` call, two on a
pump generation", put the compile seconds a call in `GenerationCost` (it has them) and let G2 report modules a
generation. The alternative, holding the refill's rows unscored until the next generation, would put fresh rows
into a tournament with NaN fitness, which the engine does not allow (4795).

### 3. What fills the pump's refill in regime P (F5, 5a.6 question 2)

With intake 16: `pump_keepers_at_most(16)` = 3 keepers, 13 refills a pump (engine.rs 3829-3831); at 2,000
evaluations, about 80 generations of 24, the pump every 5 generations refills about 16 x 13 = 208 rows, a tenth of
the budget. The deprecated pump offered at most 4 e-graph variants of the best, and only when the best had changed
(futhark_evolve.rs 3055-3081, `variants(.., 2, 16, 4)`), plus the fold's forms. The hook "caps what it returns and
the rest stay the random draw" (5362-5366). A random 47,999-word histogram gene in regime P is a wasted
evaluation (the plan's own section H: random programs crowd at the all-zero output). F5 must say what the other
nine or so rows are. Recommendation: after the variants and the fold's forms, mutants of the champion island's
best by the run's mutation operator, no two with one text or one tree, as `fable_mutants_population` already
builds them (`futhark_rings::mutants`, 1588-1631); in regime R the remainder may stay random. Every refill row that
is a new text is a budgeted evaluation, and the card should say how many a pump may spend.

### 4. Task sizes and the DAG (5a.3)

- F2 moves `Judge` (futhark_evolve.rs 457-1280, about 820 lines with the k-means paths), `Genetics`, the controls
  and the start, behind an owned object; F3 adds the per-generation prefetch, scoring and stage logic (2781-2857).
  Neither is "a few hours". Split F3 into F3a (the answer-scored problems, `prefetch` + `evaluate`) and F3b
  (k-means, `prefetch_kmeans` + `evaluate_converged` + the test case), F3a first, each with its plumbing test.
- F1 names `Host<'a>` and `futhark_sig`; `Interp<'a>` (futhark_evolve.rs 462, held by the judge) and `Genetics<'a>`
  (1283-1287) borrow too. `Box<dyn FnMut>` is `'static`, so every borrow must go; `Arc` is the right fix, never
  `Box::leak`.
- F6 depends on F3, not only F2: the k-means in-run A/A runs the reference program as a candidate through
  `prefetch` + `evaluate` (2583-2595), and the champion's re-time reads the timing slots F3 owns.
- P1 should carry G1's layout cost: the k-means layout at 14,359 types took minutes on the host (the plan's 488 s
  figure is for the larger view); P1's host test "check_layout for all six problems" will be slow for k-means;
  mark it `#[ignore]`-free but long, and time it.
- G1 before P2's GPU test for k-means: the walk stage's `binding_limit()` sets the page size
  (shape_typed.rs 316, `with_parts_paged`), and P2's mock test at 110 MB a row is the first time the paged store
  holds gigabytes.

### 5. The regression criterion is weak at three against three (5a.4)

"Within the deprecated runner's range or better" over three random seeds a runner cannot separate two runners;
three draws define a wide range. Keep the table as the result and say so in the plan: the proof that phylu
REPRODUCES the deprecated runner is the plumbing test (bit for bit) plus an anchor on the GPU path that the plan
does not yet have: **R0**, regime P generation 0 on `--eval rust`, histogram, both runners scoring the same 24
programs (F5's test gives the same programs in the same order) and writing the same quality and speed inputs bit
for bit in their ledgers. That is what makes R1's comparison a comparison of search dynamics and nothing else.
Then report R1/R2 as the measured table, six numbers a measure, difference in percent, and no verdict word beyond
"produced by both" and "worse / not worse than the deprecated runner's worst". The rule "a worse result blocks X1
until its cause is found" stands.

### 6. The memory statement in `docs/KMEANS_FITNESS.md` contradicts the measured limits (5a.6 question 1)

`KMEANS_FITNESS.md` 454 says a population of 64 k-means genes (7 GB) is "impossible on the device". That was
written against the default 128 MiB binding. The typed device is requested with the adapter's limits
(`shape_fit.rs` 100-104, `typed_device.rs` 950-951), and this machine's adapter reports
`max_storage_buffer_binding_size 4,294,967,292 bytes, max_buffer_size 4,294,967,295 bytes`
(`docs/reports/2026-10-22_ST9_volumepassage_champion/timing_151/log.txt` line 8), with 128 GB of unified memory
(`hw.memsize` 137,438,953,472, Mac15,8). The store is paged at the binding limit (`with_parts_paged`). Correct the
line once G1 has measured.

### 7. Vocabulary (the plan text)

The plan calls the reference program "the seed" throughout 5a.1 and 5a.2 ("the seed's entry", "the seed's run",
"the SEED's step") and a k-means iteration a "step". Andrew's vocabulary: reference program, iteration. The code's
identifiers can stay; the plan's prose should change.

### 8. Smaller faults

- The engine's comment promises an external fitness "checks the run file before every individual" (engine.rs
  8880-8881); the deprecated runner checks it once a generation (2770). A k-means generation is minutes:
  `evaluate_generation` should check `killme.trigger` between batches and return the rows scored so far with a
  stop flag, and P4 should end the run as `killed`.
- `evaluate_external` reads the whole population back on every call (`read_generation`, 4785): 2.65 GB for 24
  k-means rows, twice on a pump generation. Acceptable beside a k-means evaluation, but G1 should time it, and the
  evaluator should express from the head cells it needs rather than copying every row's text eagerly.
- The engine's snapshot is written on `progress_every`; the Futhark runner wrote one a generation. P4 sets
  `progress_every = 1` and says so in the card.
- `External.rates` is per island; P3's expressed-cell rate must be stated per regime, because the deprecated
  runner mutated 1-2 expressed cells in P and 1-4 in R (`m_of`, 3156). The device's inversion and crossover act on
  the whole 47,999-word row, so on this kingdom they are nearly always neutral; marriage is the one structural
  operator, which is why P3 belongs before R1.
- 5a.2's `RowScore.fitness` is `f64`; the device reads `f32`. Fine at these angles (0.7 rad, d = 66), say so.
- `GenerationScores.evaluations` should be the run total AND the call's new evaluations, so P2(a)'s arithmetic is
  not a subtraction phylu does.

## Recommended changes to the plan, concrete

1. Reorder the critical path: F1 -> F2 -> F3a -> P1 -> P2 (with the widened `ExternalFresh` type defined in F2's
   module) -> G1 -> F3b -> P3 -> F4..F7 -> P4 -> G2 -> G3 -> R0 -> R1.
2. Rewrite 5a.1's row "batch evaluation" to "one module an `evaluate_generation` call; two on a pump or fold
   generation; k-means's test case a third"; add `modules_a_generation` to G2's expectations.
3. F5: specify the refill's content per regime (variants, fold forms, then mutants of the champion island's best;
   random only in R) and its budget share in the card.
4. P2(e): place the hook's genomes in EVERY pair's intake, not the first; P2(a): the hook returns the charged
   count; P2(b): the hall of fame restarts and the `stage_change` event is written; P2(d): a `serde_json::Value`
   slot in `Checkpoint` (format version 4).
5. 5a.4: add R0 (generation 0 on the GPU, bit for bit) and state the three-against-three limit.
6. Split F3; add `Interp<'a>` to F1; make F6 depend on F3.
7. Replace "seed" by "reference program" and "step" by "iteration" in 5a's prose; correct `KMEANS_FITNESS.md` 454.

## The four open questions

**(a) k-means on the device.** It fits. One gene at N = 192 is 27.6 million words = 110.4 MB; 24 rows are 2.65 GB
a side, 5.3 GB with both sides, 4 % of the 128 GB of unified memory. The adapter's limits are requested, and they
are 4 GiB a binding and a buffer (measured, finding 6), so a side is one page of the store (or a few, if the walk
stage's `binding_limit()` is smaller); two pairs (48 rows, 10.6 GB) would fit as well. Metal's
`recommendedMaxWorkingSetSize` is recorded nowhere in either repository: G1 should print it beside the init,
variation and read-back seconds and the walk stage's per-page output size, and `KMEANS_FITNESS.md` 454 is then
corrected. Keep N = 192 (the reference program is 131 functions) and 24 individuals for the regression; the bound
on the population is the evaluation cost (a whole k-means run a new text, 13.2 s on the c backend), not memory.

**(b) The island shape.** One pair, intake 16, champion 8, pump every 5, the engine's default cohorts
(`cohort_merge` = 5 x 5 = 25 generations) is the right like-for-like against a population of 24. Two numbers to
write down: `promote_fraction` 0.03 of 8 rounds to 0 and the floor makes it 1 promotion a beat; the refill is 13
rows a beat (finding 3). Keep 24 for R1-R4; the A/B after R1 is two pairs of (16 + 8), which the device holds.

**(c) Operators.** Agreed: the device's point mutation at an expressed-cell rate, its crossover and inversion, plus
marriage, for the regression; the row operators (P3b) as an A/B after R1, host twin first. State the expressed-cell
rate per regime (finding 8) so R1 mutates as the deprecated runner did.

**(d) Regime C.** Random search has no selection, so it is not evolution; but it is a budgeted loop over
generations of fresh individuals with a ledger, a hall of fame and a stream (futhark_evolve.rs 3029-3039,
3411-3416), which is a runner, and the ruling says fuller keeps none. Put it in phylu as a control arm of
`futhark_fit` (`--regime C`): `pop` fresh individuals a generation from `FutharkEvaluator::fresh(Random)`, scored
through the same `evaluate_generation`, the same outputs, no engine. It then lives beside the run it controls,
runs on the same binary and evaluator, and runs AFTER the interesting run (the ruling's order). `futhark_measure`
keeps what scores a fixed list of programs: the slope check, the operator measurement, the rescore, the A/A sets.
