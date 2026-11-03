# Review: the flat k-means fitness as built, and the first run (advisor, round 3)

*fuller, 2026-11-03 16:10Z, branch `feat/futhark-kingdom` at f46d424a (the working tree clean but for the untracked
`src/futhark_reader/kinds.rs`). Read in full: rounds 1 and 2; the commits be0ae82e, 7e9701c0, 1eb25cbb, f46d424a;
`docs/KMEANS_FITNESS.md`; `src/fitness.rs`; `src/futhark_sig/kmeans_converged.rs` and its tests; `tests_flat.rs`;
the k-means parts of `src/futhark_batch.rs` and `futhark-rust-rt/src/host.rs` with `kmeans_harness.rs`; the converged
judge of `examples/futhark_evolve.rs`; the lead's `kmeans_gpu_controls.sh`, `kmeans_run_s1.sh` and their logs as
they were written. Read-only; the measurements are one Python script over the stored cases (round 2's harness, the
Rust port's arithmetic to the printed digit) and the lead's own GPU logs. Nothing of mine touched the GPU or built
anything; the lead's controls were running throughout.*

## Verdict

The scheme as built is the flat vector Andrew ruled and nothing else: `fitness::qqs` repeats the quality group m
times, appends the speed group once, and the HFF angle over that is the fitness; no default scheme carries the offset,
and the gated names are registered with their fault. The agent's ordering table reproduces to four places. The
resident sample is uploaded once a session and found resident by every later run (the GPU test: 1 upload, then 1
resident hit); the runtime's harness is fuller's to the bit; the compiled seed on the GPU converges on every fitness
case at the same iteration as `futhark c`, its quality inputs exactly 1/2 against itself.

What is not solved is the early stop, and the measurement says it is not a corner: a thresholded Lloyd step (a real
gene, paying the full step cost every call) that stops when no coordinate moves by more than 3e-3 reaches 99.7 % of
the seed's gain in two thirds of the iterations and ranks above the seed at every m up to 143. Under the committed
inputs, "stop early" is the problem's exchange rate, and the first run will most likely find it. One input measured
below separates it from a genuine converger: **the iterations UltraDim's own rule still needs from the candidate's
stopping point**, entered as a quality input (so repeated m times) through the same `r/(1+r)` against the seed's. With
it, every early stop and every thresholded step ranks below the seed at every m from 1, the same-partition
reorderings and the faster step stay above, and the price is that a step that stops at a point which is not a
Euclidean fixed point (the normalised step; the step rounded to 1e-3) pays about 10 % of fitness. Andrew decides;
the numbers are in section 2.

Go for the first run as scripted, with two things said plainly: the run measures the committed scheme, and its
champion is to be read with the early-stop table beside it (the per-generation line prints the four inputs, so a
champion at speed 0.25 and quality 0.505 is an early stop, not a better k-means); and the two paths (`--eval single`,
`--eval rust`) score an unconverged candidate on different assignments (N1 below), which does not touch this run's
converged candidates but must be fixed before any result is compared across backends.

## 1. The scheme as built

- `fitness::qqs` (`src/fitness.rs` 570-582): the range check, `flat_vector` (the quality slice `quality_weight`
  times, then the speed slice), `truenorth_angle` over it; the two group angles beside it are reported, never read by
  selection (`evaluate`, `futhark_evolve.rs` 1157-1170: `fitness: f.fitness`). `Scheme::converged_fitness` (440-452)
  routes `FutharkConvergedQqs` to `qqs`; `staged_weighted` (593-607) routes `StageTwo::Flat` to `qqs`. The judge
  builds the quality group once per case (`evaluate_converged` 1121-1126); the repetition happens inside `qqs` only, so
  the ledger's `quality_inputs` column and the per-generation line show each input once, as they should.
- The offset survives only in `converged_gated` (535-548) and `stage_two`'s `Gated` arm (618-619), reached only by the
  four names that carry `fault: Some(OFFSET_FAULT)`; `default_for` names a flat scheme for every family (358-366);
  `Problem::fitness` is `futhark-kmeans-qqs` for k-means and `futhark-slope4-qqs` for the others. No residue.
- The run card, the header and `run_start` record `quality_weight` (`start_weighted`, 482-484; the runner 2141-2160,
  2310); the objective names list each quality column m times (2291-2296).
- One wording fault: the specification and the module say each side's seconds is "the median of its runs"
  (`converge_interleaved` 583-586, `sides_of_runs` 603-606): `v[v.len() / 2]` of two sorted values is the LARGER. At
  the default `--pairs 2` each side's seconds is its slower run. Harmless for the A/A (both sides alike), but it is
  not a median, and the first run's seconds inputs are maxima of two. Three pairs give a median; or say "the slower".

## 2. The early stop (item 2): re-measured, and an input that separates it

The agent's table (`the_flat_scheme_at_m_2_and_3_...`, 500 rows k 8) reproduced by the Python harness under the same
cost model (a seed-work step costs the seed's seconds an iteration, a constant step costs 0, the faster step half):
seed 0.7227; identity 0.6571 / 0.7556 / 0.8011 at m = 1 / 2 / 3; worsening 0.9140 / 1.0574 / 1.1247; NaN 1.5708;
rounded 1e-3 0.7229; faster-same-bits 0.6685 / 0.6870 / 0.6961; one seed step then identity 0.5463 / 0.6129 / 0.6439.
All as stated.

**Is "one step then stop" a winner this fitness will find?** Yes, and not only as the free identity. A gene cannot
count iterations, but it can test the size of its own move: the step below computes the Kahan means and returns its
input unchanged when no coordinate moved by more than τ (a `sel` on the change), and pays the seed's cost every call.
Measured (`early_stop.py`), 500 rows k 8, the committed scheme:

| step | iterations | quality [cos, pct] | speed [s, it] | m = 1 | m = 2 | m = 3 | ranks below the seed from |
|---|---:|---|---|---:|---:|---:|---|
| seed | 9 | 0.5000, 0.5000 | 0.5000, 0.5000 | 0.7227 | 0.7227 | 0.7227 | - |
| thresholded Lloyd τ 1e-3 | 9 | 0.5000, 0.5000 | 0.5000, 0.5000 | 0.7227 | 0.7227 | 0.7227 | equal to the seed |
| thresholded Lloyd τ 3e-3 | 6 | 0.5008, 0.5005 | 0.4000, 0.4000 | 0.6523* | 0.6768* | 0.6888* | m > 143 |
| thresholded Lloyd τ 1e-2 | 3 | 0.5056, 0.5041 | 0.2500, 0.2500 | 0.5711* | 0.6279* | 0.6547* | m > 38 |
| one seed step, then identity (free) | 3 | 0.5056, 0.5041 | 0.1000, 0.2500 | 0.5463* | 0.6129* | 0.6439* | m > 44 |
| four seed steps, then identity (free) | 6 | 0.5008, 0.5005 | 0.3077, 0.4000 | 0.6249* | 0.6592* | 0.6758* | m > 196 |

At τ 3e-3 the step leaves 0.3 % of the seed's Euclidean gain on the table (141.04 against the fixed point's 140.65,
from a start of 268.57) and saves a third of the iterations; no repetition under 143 ranks it below the seed. Point
mutation reaches such a step from the seed in a handful of edits (a `sel` on `sub`); evolution will find it or its
equivalent, and the champion will be a slightly worse k-means that stops sooner.

**What distinguishes it, in the measures.** Continue UltraDim's own rule (the seed's step, its tolerance 1e-4, its
cap) from the candidate's final centres and count the iterations it needs to stop: **L**. The seed, the
same-partition reorderings, the damped and the over-relaxed steps end at a fixed point (L = 1: one seed step leaves
the inertia unchanged); the identity needs 9, the early stops 5-8, the thresholded steps 5-8, the rounded and the
normalised steps 2 (their final centres are not Euclidean means). Three ways to enter it, measured:

| input | definition | seed | identity | τ 3e-3 | τ 1e-2 | one step (free) | normalised | rounded 1e-3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | `squashed_ratio(L_cand, L_seed)`, in the quality group | 0.5000 | 0.9000 | 0.8333 | 0.8889 | 0.8889 | 0.6667 | 0.6667 |
| P2 | `left` = (I_final − I_fix) / (I_start − I_fix), absolute, 0 best | 0 | 1.0000 | 0.0030 | 0.0230 | 0.0230 | 0.0909 | 0.0005 |
| P3 | `squashed_ratio(left_cand, left_seed)` (the seed's is 0) | 0 | 1 | 1 | 1 | 1 | 1 | 1 |

The flat HFF with each, m = 1 / 2 / 3 (`*` above the seed):

| step | committed | + P1 | + P2 | + P3 |
|---|---|---|---|---|
| seed | 0.7227 | 0.7227 | 0.6435 / 0.6224 / 0.6126 | 0.6435 / 0.6224 / 0.6126 |
| identity | 0.6571* / 0.7556 / 0.8011 | 0.8347 / 0.9355 / 0.9789 | 0.8848 / 0.9933 / 1.0401 | 0.8848 / 0.9933 / 1.0401 |
| thresholded τ 3e-3 | 0.6523* / 0.6768* / 0.6888* | 0.7998 / 0.8485 / 0.8700 | 0.5813* / 0.5832* / 0.5841* | 0.8818 / 0.9463 / 0.9746 |
| thresholded τ 1e-2 | 0.5711* / 0.6279* / 0.6547* | 0.7741 / 0.8508 / 0.8839 | 0.5096* / 0.5418* / 0.5558* | 0.8325 / 0.9186 / 0.9559 |
| one seed step, then identity | 0.5463* / 0.6129* / 0.6439* | 0.7590 / 0.8420 / 0.8777 | 0.4876* / 0.5289* / 0.5467* | 0.8182 / 0.9103 / 0.9500 |
| faster, same bits | 0.6685* / 0.6870* / 0.6961* | 0.6797* / 0.6961* / 0.7034* | 0.5957* / 0.5919* / 0.5902* | 0.5957* / 0.5919* / 0.5902* |
| plain f32 sum (same partition) | 0.7227 | 0.7227 | 0.6435 / 0.6224 / 0.6126 | 0.6435 / 0.6224 / 0.6126 |
| normalised centroids | 0.7228 | 0.7798 / 0.7935 / 0.7997 | 0.6463 / 0.6260 / 0.6165 | 0.9273 / 0.9734 / 0.9939 |
| rounded 1e-3 | 0.7229 | 0.7798 / 0.7936 / 0.7998 | 0.6436 / 0.6225 / 0.6127 | 0.9274 / 0.9735 / 0.9940 |
| rounded 1e-2 | 0.7859 / 0.7701 / 0.7620 | 0.8797 / 0.8913 / 0.8965 | 0.6998 / 0.6636 / 0.6465 | 0.9699 / 1.0023 / 1.0168 |

- **P2 does not separate.** The gain k-means makes is front-loaded: the early stops leave 0.3-2.3 % of it, and an
  input of 0.02 in the quality group buys nothing against speed inputs of 0.25. The exchange rate stays at m ≈ 37-143.
- **P3 is the retired gate's cliff in another form**: the seed's own value is 0, so any candidate not at the bit-level
  fixed point scores 1 (the reorderings happen to land exactly on it here; a partition one point different would not).
- **P1 separates at every m from 1**: every early stop, free or paying, ranks below the seed (0.84-0.85 at m = 2
  against 0.7227), the reorderings and the faster step keep their place. The cost: L is a count, so a stop that is
  not a Euclidean fixed point pays the step from 0.5 to 0.667 whatever its distance from one; the normalised step
  (round 1's F8, the finding evolution "may find") reads 0.7935 at m = 2 instead of 0.7228, as does rounded 1e-3. That
  is the criterion saying "UltraDim's stopping rule is Euclidean, and the run is scored at a point that rule would
  stop at", which is what the specification already states; it is a graded input, not a gate, and `kmeans_inputs`
  stays total on [0, 1]. Its cost is one seed run from the candidate's end a case (a few seed steps, ms on the GPU;
  the seed's entry is in every module already), with the seed's L measured the same way (1 on every case here).
- Entered in the speed group instead (once, not m times), P1 also ranks every early stop below the seed at every m
  (the free one-step adds 0.54 to the sum of squares against the 0.43 it saves; τ 3e-3 adds 0.44 against 0.18), with
  the same 10 % on the normalised step; the quality group is the right place because "did it finish" is a statement
  about the clustering, and its repetition is what Andrew's m is for.

The pooled-over-cases view is the same: the run's vector is six cases' inputs, and the early stop's margin is per case.

**The held case says the same** (`held_n1000_k32`, 1,000 rows k 32, the seed 10 iterations, I 483.46 → 261.72):

| step | iterations | left | L | committed m = 1 / 2 / 3 | + P1 |
|---|---:|---:|---:|---|---|
| identity | 2 | 1.0000 | 10 | 0.6469* / 0.7448 / 0.7900 | 0.8325 / 0.9338 / 0.9774 |
| rounded 1e-3 | 9 | 0.0006 | 2 | 0.7032* / 0.7098* / 0.7131* (above to m = 286) | 0.7651 / 0.7845 / 0.7932 |
| normalised centroids | 12 | 0.0897 | 2 | 0.7579 / 0.7463 / 0.7404 | 0.8064 / 0.8099 / 0.8116 |
| plain f32 sum (same partition) | 10 | 0.0000 | 1 | 0.7227 | 0.7227 |
| faster, same bits | 10 | 0 | 1 | 0.6685* / 0.6870* / 0.6961* | 0.6797* / 0.6961* / 0.7034* |
| one seed step, then identity (free) | 3 | 0.0358 | 9 | 0.5429* / 0.6120* / 0.6441* (to m = 32) | 0.7627 / 0.8481 / 0.8849 |
| thresholded Lloyd τ 3e-3 | 8 | 0.0009 | 4 | 0.6823* / 0.6961* / 0.7029* (to m = 321) | 0.8053 / 0.8426 / 0.8591 |
| thresholded Lloyd τ 1e-2 | 4 | 0.0153 | 8 | 0.5868* / 0.6365* / 0.6602* (to m = 57) | 0.7839 / 0.8558 / 0.8870 |

Two things to note from it. The step rounded to 1e-3 is itself an early stop on this case (9 iterations against 10,
0.02 % worse) and beats the seed under the committed scheme at every m up to 286; P1 reads it as it is (L = 2). And
the plain f32 sum lands one f32 ulp above the fixed point (I_final 261.7200 against I_fix 261.7199, left 4e-7): P3
scores it 1, the round-2 cliff reproduced exactly; P1 reads L = 1 because the seed's rule stops within its tolerance.

## 3. The i32 problems' default `futhark-slope4-qqs` at m = 8 (item 3)

- The slope check scores stage 1 only: `slope_check_command` (`futhark_evolve.rs` 1848, 1851) asks for
  `FitnessStage::Quality`, whose branch in `staged_weighted` (`fitness.rs` 595-601) is TrueNorth over the quality
  inputs alone and never reaches `qqs`, `Gated` or `Ungated`. So the check's verdict on the stored outputs is the one
  recorded for the quality parts, unchanged by the flat stage 2: histogram (15 cases) pass, pathfinder (19 cases)
  pass (`docs/measurements/2026-10-29_futhark_slope_check.tsv`), A7 pass at 0.9667 (`2026-11-03_futhark_a7_slope_check_cpu.tsv`).
- The ranking at m = 8 is `tests_flat::an_inexact_program_ranks_below_the_exact_one_at_the_recorded_weight`: 80
  one-edit neighbours of the published gene on each of histogram and pathfinder, each inexact one given every timing
  slot 0, the exact program every slot 1/2, through `qqs` itself; the least m is `ceil((11/4) / min Σq²)` and the
  recorded constant must be at least it (pathfinder's least sum 0.37 gives 8, histogram's 31.7 gives 1). The test is
  pooled over the fitness cases, which is the run's own vector, not a per-case claim; a neighbour inexact on one case
  only still carries that case's inputs. Run here from the lead's built test binary (CPU, 10.8 s): histogram 79
  inexact neighbours of 80, least sum of squares 31.72, least m 1, at m = 8 the exact program 0.0975 against the best
  inexact 0.9744; pathfinder 79 inexact of 80, least sum 0.3666, least m 8, **at m = 8 the exact program 0.0872
  against the best inexact 0.0901**: a margin of 3 %, the least integer and no more. Every inexact one-edit neighbour
  ranks below the exact program at m = 8 on both problems; the ordering holds, but on pathfinder it holds by the width
  of one rounding of the timing slots (a neighbour whose slots read 0.03 instead of 0 would still be below; one that
  is also inexact by less on a two-edit path would not be covered by the test). If a margin is wanted, m = 10 costs
  nothing the test does not already accept (N6).

## 4. The resident-sample path (item 4)

- **Uploaded once, kept for every generation and candidate.** `Session::resident` (`host.rs` 97-151) is a buffer the
  session owns, replaced only by a sample of another fingerprint; `Host` holds the one `Session` for the run
  (`futhark_batch.rs` 265, 1043-1047), and every `k` command reads it through `RunState::resident`. It is dropped on a
  device failure or a panic in the generated module and uploaded again with the next context (host.rs 546-562,
  596-600, 617), which the `u` record reports. The GPU test measured it: 1 upload (0.005 s), then 1 resident hit
  across two runs and two modules.
- **The harness on the host is fuller's to the bit:** `kmeans_harness.rs` `assign` and `euclidean_inertia` are
  fuller's text (threads over chunks in place of rayon, each point's sums in order), and
  `the_runtimes_harness_is_fullers` includes the file by path and asserts both functions equal fuller's on the dbpedia
  rows against the starts and the seed's centroids, inertia by bits. The device's assignment is never read: the loop's
  assignment and stopping rule are the host's on the centres read back, as `converge` does on the c backend.
- **The GPU controls log** (`/private/tmp/claude-501/fbench/kmeans_gpu_controls.log`, read at 16:05Z): step 1, the
  GPU test, passed in 12.4 s (three cases, GPU and `futhark c` at the same iteration, cosine inertia equal to five
  places, quality inputs exactly 1/2; the candidate side's seconds 0.61 / 0.57 / 0.50 against the seed's at one pair:
  the first entry called pays its cold start inside the timed passes, which is why two pairs are the floor). Step 2
  (`--eval single`, a process an iteration): the seed on all six fitness cases in 10.3 s (the program's own 0.08-0.42
  s a case, 1.1-2.4 s wall); the in-run A/A at 2 pairs 40.9 s, fitness 0.7205, **quality inputs exactly 1/2 on
  every case (the GPU reduction is deterministic)**, speed inputs 0.451-0.5035, floor 0.049. Then the e-graph's 31
  variants at 15:43Z; generation 0 at 15:46:44Z, 40.9 s for its 2 evaluations. Step 3 (`--eval rust`, the resident
  sample, finished 15:55:24Z): the seed's module compiled in 1.2 s, the sample uploaded in 0.005 s (30 MB), the seed's
  6 cases in 4.6 s (10.3 s before); the in-run A/A 6.4 s (40.9 s before), fitness 0.7237, quality inputs exactly 1/2
  on every case, speed inputs 0.4992-0.5232, **floor 0.0232**; the e-graph's 31 variants; then generation 0's batch:
  **23 programs in one module, compile 2.1 s, upload 0 (1 resident hit), run 129.6 s, 135.5 s in all** = 5.6 s a
  program at 2 pairs on 6 cases, 10.6 evaluations a minute against 1.5 on the process path. The champion on the test
  case: the seed's own clustering (inertia 0.14079, 11 iterations, seconds 0.211 against 0.202), reported, not
  scored. The run script started at 15:55:24Z on f46d424a's binary. At 135 s a generation of 24 and the 8-minute
  start, the 60-minute run is about 23 generations, 550 evaluations.

## 5. The run script (item 5)

`kmeans_run_s1.sh`: population 24, elites 2, `--ops point=0.5,crossover=0.2,marriage=0.1,invert-rows=0.1,transpose-rows=0.05,cross-rows=0.05`
(sums to 1, every name in `OPS`), `--pump-every 5`, `--pairs 2`, `--quality-weight 2`, `--eval rust --backend gpu`,
`--evaluations 100000 --minutes 60`, seed 1, from the published seed. It is the right first run: the committed scheme
at Andrew's m, the resident path measured first by the controls, one seed. Two notes:

- The clock is the stop (100,000 evaluations is never reached), and the P start spends the e-graph (149 s) and the
  saturated fold (193 s, 19.5 GB) before generation 0: about six of the sixty minutes. Fine on this machine (137 GB),
  but the report must count generations, not minutes, when runs are compared.
- What the per-generation line shows (`futhark_evolve.rs` 2905-2940): the best's HFF and, per fitness case, its four
  inputs against the seed (`q [cos pct] s [seconds iterations]`, 1/2 the seed), and the champion on `test_n1000_k32`
  with its inputs and both sides' raw measures (inertia, log10 p, seconds, iterations), reported, never scored; the
  stream's extras carry `upload_s` and `resident_hits`; the report carries `seed_aa_speed_floor`,
  `best_seconds_margin_below_half`, `speed_claim_beats_noise_floor`, `test_case`. That is what the brief asks. What is
  missing: the best's RAW measures on the fitness cases (its iterations and seconds beside the seed's) are not printed
  per generation, only its inputs; the test-case line has them. One more line, so an early stop is read as "3 of 9
  iterations", costs nothing.
- The run stops on the seed's side failing in the host (`kmeans_batch`, 1015, 1019-1021: "our software has a bug").
  Right by the house rule; but on a device a candidate's kernel can lose, the seed's command after it runs on a fresh
  context (host.rs 546-562), so the stop fires only if the seed itself fails, which is the right trigger.

## 6. New findings

- **N1. The two evaluation paths score an unconverged candidate on different assignments.** `converge`
  (`kmeans_converged.rs` 511, 521, 541) measures the final centroids under the assignments made BEFORE the last step
  (UltraDim's loop: assign, update, `calculate_inertia` on those assignments); `run_of_k_payload` (873) assigns the
  points to the FINAL centroids and measures under that. At a fixed point they coincide (bit-identical centres), so
  the seed, the A/A and every converged candidate read the same on both paths, and the GPU test could not see it; a
  candidate stopped by the cap (100 iterations, "slow, not worst") or by a sub-tolerance move is scored on different
  partitions by `--eval single` and `--eval rust`. The comment at 871-872 says the two read "the same way"; they do
  not. The fix is small: the runtime has the assignments it stopped on (host.rs 249) and can write them after the
  centres (n × 4 bytes), or the payload can carry the penultimate centres. Until then the two paths are two measures.
- **N2. "Median" of two is the maximum** (section 1): at `--pairs 2` each side's seconds is its slower run.
- **N3. Thirty megabytes parsed per `k` command.** `kmeans_loop` (host.rs 220, 231) converts the whole resident
  sample from bytes to `f32` on every command (5,000 × 1,536 floats) to slice the case's rows: 2 pairs × 2 sides × 6
  cases × 24 programs = 576 conversions a generation, each a 30 MB allocation. Not a fault of the measure (the
  program's time is the GPU's passes); seconds of host time a generation, and a generation's wall clock is the budget.
  Convert once a run, or slice the bytes.
- **N4. The seed's L is assumed 1.** If P1 is adopted, the seed's own L on each case must be measured at the start
  (the control) and written beside its iterations; the harness's tolerance makes it 1 on every stored case here, and
  a case where it is not would be a non-deterministic reduction, which the A/A already catches.
- **N6. `ANSWER_QUALITY_WEIGHT` = 8 is the least integer with no margin on pathfinder** (section 3: 0.0872 against
  0.0901 at m = 8). The constant is the measured least, as its comment says; a run that wants an inexact program never
  to win on speed should take `--quality-weight 10` or more, and the test's claim covers one-edit neighbours only.
- **N5. The docs index.** This file makes `docs/index.html` stale and the index test red until
  `python3 docs/build_index.py` is run with its write option; this review commits only its own file (as rounds 1
  and 2 did), so the agent regenerates the index with the next commit.

## Dispositions of rounds 1 and 2 still open

| item | disposition |
|---|---|
| round 2, the flat scheme (Andrew's ruling) | built as ruled: `qqs`, m a run parameter, default 2, both m = 2 and 3 in the kingdom's test; the gated names retired with the fault |
| round 2, F1 / N1 (the red test) | fixed in 2e646002 (`squashed_ratio` returns 1 on a NaN on either side; the seed gain 10) |
| round 2, N2 (the test case never run) | fixed: `converged_on_test` every generation the champion's text changes and at the end; `OUT/test_case.tsv`; the report's `test_case` |
| round 2, N3 (the noise floor) | fixed: the in-run A/A on every backend, `seed_aa_speed_floor`, `speed_claim_beats_noise_floor`; measured on the GPU single path 0.049 (section 4) |
| round 2, N4 (one pair is no interleaving) | fixed: default 2; but see N2 above (the "median" of two) |
| round 2, N5 (the P start's cost) | measured: 149 s + 193 s, 19.5 GB on the seed |
| round 2, the early stop beats the seed up to m = 32-44 | still true of the committed scheme and worse than stated: a paying thresholded step wins to m = 143; section 2 proposes P1 |
| round 1, F8 (Euclidean loop, spherical score) | stated in the specification; P1 would make it binding (a non-Euclidean fixed point pays) |
| round 1, F10 (the pump's cost) | measured (N5 above) |
| round 1, the Reader reads none of the 25 implementations | unchanged; `kinds.rs` still untracked. The pool is the seed and the textbook step; the paper's 25-implementation protocol is not run |
| round 1, item 5 (Fable's implementation as a named arm) | not built; not needed for the first run |

## Go / no-go

**Go**, and it has gone: the controls' step 3 showed the resident path's A/A (quality inputs exactly 1/2, the floor
0.0232) and a generation of 23 programs in 135 s, and `kmeans_run_s1.sh` started at 15:55:24Z on f46d424a's binary.
The reading rule: the committed scheme is what it is, the early stop is its likely winner, and the run's value is
measuring whether evolution takes that exchange or finds a faster genuine converger; the per-generation inputs say
which (a champion at `s [0.25 0.25]` with `q [0.505 0.504]` is a stop at a third of the iterations, not a better
k-means). Decide P1 (or not) from section 2 before the second seed, not during this run; a seed-2 run with P1 beside
a seed-2 run without it is the comparison that settles it.

## Scripts and logs

`/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/1a542778-6355-45e6-9c87-a491851d5eb3/scratchpad/km3/early_stop.py`
and `early_stop.log` (the harness imported from round 2's `km2/measure2.py`); the lead's logs in
`/private/tmp/claude-501/fbench/` (`kmeans_gpu_controls.log`, `kmeans_gpu_before.log`, `kmeans_gpu_after.log`,
`kmeans_resident_gpu_test.log`).
