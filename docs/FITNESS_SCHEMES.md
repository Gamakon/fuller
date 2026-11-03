# Fitness schemes

Every fitness function an experiment in fuller or phylu has selected on, by name, with the runners that take it and the commit of each past experiment.

## Context

- **Objective** (Andrew, 2026-11-01): "Fitness is a configurable definition, set out per experiment. The tooling supports new fitness schemes and all old ones."
- **Approach.** A fitness function is a named scheme in `fuller::fitness::SCHEMES` (`src/fitness.rs`). A runner takes `--fitness NAME`, or `EVOLVE_FITNESS=NAME` for phylu's env-configured regex engine. With no name, it uses its family's default, which is the newest scheme. The scheme is recorded in three places: the header line (`# fitness <name> (<commit>): ...`), the stream's `run_start` (`"fitness": {name, family, commit, description}`) and the report or run card.
- **Rule.** A new fitness is a new scheme beside the others. It never changes what an existing name computes.
- **Old schemes.** These are kept so that an earlier experiment's definition stays nameable and runnable. Each one is marked **NOT TO BE RUN AS IS**: it has a known fault, and the fitness function must be redefined before it is used. A runner given an old scheme prints the warning and the fault in its header. To reproduce an old result exactly (cases, timing protocol and all), check out the experiment's commit, listed below.

## The schemes

| name | family | commit | definition | status |
|---|---|---|---|---|
| **`futhark-kmeans-quality-first`** | futhark_sig (k-means only: `Problem::fitness`, `fitness::scheme_for_problem`) | fuller, the review of 2026-11-03 round 1 | The candidate and the seed are each run to UltraDim's stopping rule (assignment, update, Euclidean inertia change < 1e-4; the 100-iteration cap is UltraDim's `converged: false`, a slow finished run) on every fitness case, interleaved `--pairs` times (default 1), the inputs against the seed's run of that interleaving, each r/(1+r) of a ratio, total on [0, 1] (1/2 is "equal to the seed"): quality = the cosine inertia of the final centroids; the seed's mean log10 angular percentile over the candidate's (the points off their centroid; UltraDim's number over every point reported beside it); speed = the program's own time to the stopping point (the GPU-timed passes, or `futhark c`'s `-t`); the iterations. A run that does not finish is 1 on all four. One stage: the two-level HFF of the two groups, gated by quality (a candidate worse than the seed on any quality input scores π/2 + θ_quality). The work guard is 10 x the seed's work a case. (`docs/KMEANS_FITNESS.md`, `futhark_sig::kmeans_converged`, `fitness::converged_gated`.) | default for k-means |
| `futhark-kmeans-converged` | futhark_sig (k-means) | fuller bd929652 | The same loop; per case, five inputs relative to the seed's run at start-up: cosine inertia; 10^(difference of mean log10 p over every point); the harness's wall clock to the stopping point; iterations; the seed's inertia gained an iteration over the candidate's. One stage, no gate: TrueNorth over every case's inputs. | **NOT TO BE RUN AS IS**: a candidate that gains nothing (the identity step) has a NaN rate input and a worsening one a negative input, and the run stops on either (`fitness::converged`); four of the five inputs reward the do-nothing step; three reward fewer iterations and two are one number; the percentile input is dominated by points at their centroid; the clock is the harness's (`docs/reviews/2026-11-03_kmeans_plan_advisor_round1.md`, F1-F5). |
| **`futhark-slope4-gated`** | futhark_sig (the problems with stored answers: reduce, scan, histogram, pathfinder, a7) | fuller f465a5cf | Quality, per case: the wrong share; Spearman and Pearson disagreement; the shares off by over 50 % and 1 % (`QUALITY_PARTS`). Stage 2 is gated: an inexact program scores π/2 + θ_quality. | default |
| `futhark-slope4` | futhark_sig | fuller b8163635 | The same quality inputs, with stage 2 ungated. | **NOT TO BE RUN AS IS.** Known fault: with stage 2 ungated, a fast inexact program ranks above an exact one (fixed in f465a5cf). Redefine the fitness function before using it. |
| `futhark-study7` | futhark_sig | fuller 38216485 | Quality, per case: the wrong share; the shares off by over 1, 10 and 50 %; the mean and largest error over the output's range, each e/(1+e); the order-wrong share. Stage 2 ungated. | **NOT TO BE RUN AS IS.** Known faults: the seven inputs reward the all-zeros output and fail the slope check on histogram (6f76baac, b8163635), and stage 2 is ungated (f465a5cf). Redefine the fitness function before using it. |
| **`staged-gated`** | staged | fuller f465a5cf | Stage 1: TrueNorth over the runner's quality inputs until the best is exact. Stage 2: the two-level HFF, gated. | default |
| `staged` | staged | fuller 922ec1a6 | The same, with stage 2 ungated. | **NOT TO BE RUN AS IS.** Known fault: with stage 2 ungated, a fast inexact program ranks above an exact one (f465a5cf). Redefine the fitness function before using it. |
| **`shapes-hff-gated`** | shapes | fuller f572a7cc | The two-level HFF of the log-scaled disagreements and the three cost ratios, gated. An unexpressed gene scores 4.0. | default |
| `shapes-hff` | shapes | fuller b1bccf66 | The same HFF, ungated. An unexpressed gene scores 1.5. | **NOT TO BE RUN AS IS.** Known fault: ungated, a wrong but cheap program ranks above an exact but dear one (f572a7cc). Redefine the fitness function before using it. |
| `shapes-two-band` | shapes | fuller f3972766 | A correct program scores 0.49·t/(t+t_ref); an incorrect one scores 0.5+0.5(1−agreement), in f32. An unexpressed gene scores 1.5. | **NOT TO BE RUN AS IS.** Known fault: a correct program is ranked on traffic alone, with allocations and depth unread, and this is not the runners' two-level HFF (replaced in b1bccf66). Redefine the fitness function before using it. |
| **`regex-raw`** | regex | phylu 759f0936 | The objectives are read as they are (on [0, 1] by construction). No length objective. | default |
| `regex-raw-length` | regex | phylu 759f0936 | `regex-raw` plus the rule-length objective. This is what `EVOLVE_REGEX_PARSIMONY=1` selects. | current alternative |
| `regex-colmax` | regex | phylu 8b05aaa5 | Each objective is divided by its generation-0 column maximum. No length objective. | **NOT TO BE RUN AS IS.** Known fault: the angle depends on the layout's first population, so runs are not on one scale (759f0936). Redefine the fitness function before using it. |

In stage 2, the speed inputs are the runner's timing slots, each r/(1+r).

**Why k-means is scored on its clustering objective and not against a reference (Andrew, 2026-11-03).** k-means is
not an exact computation: the centroid update is an approximate, iterative process with no unique right answer, so
agreement with a reference program (a bit-for-bit output, a membership match, a distance to f64 centres, a rounding
bound) is the wrong measure. `futhark-slope4-gated`'s bit-for-bit comparison was used on k-means until 2026-11-03 and
is **NOT TO BE RUN AS IS on k-means**: the GPU run of that day (`futhark_evolve --problem kmeans --check --eval rust`,
04:06Z) called a correct program wrong because the device adds f32 in another order than the sequential reference
(65-81 % of the cells one or two f32 bits off on 4 of 5 cases), and the fix tried (quantising the data to multiples of
2^-6, 06126f86) was reverted (fc013686): customer data has no such bound, and a compensated sum is MORE accurate than
the reference such a rule compares against. `scheme_for_problem` refuses an answer-scored scheme for k-means by
name. The i32 problems and A7 keep their schemes (A7's own f32 criterion is still open: its bit-for-bit comparison
has not run on the GPU).

## Runners

| family | runners | how to choose |
|---|---|---|
| futhark_sig | fuller `examples/futhark_evolve.rs` (every problem, A7 included since 2026-11-03: `--problem a7`) | `--fitness`; the default is the PROBLEM's (`Problem::fitness`): `futhark-kmeans-quality-first` for k-means, `futhark-slope4-gated` for the others; a scheme of the other kind is refused. The k-means scheme takes `--eval single` (each iteration its own process of the compiled plain program, its own time read from the shim's GPU timestamps or `futhark c`'s `-t`). k-means's seed is UltraDim's kernels translated (`Seed::Translated`, `docs/kmeans_seed.md`; plan 4c (d), CPU 2026-11-03). |
| staged | phylu `examples/wgsl_true_case_fit.rs`, fuller `examples/futhark_pixel_evolve.rs` | `--fitness` |
| shapes | fuller `examples/futhark_a7_evolve.rs` (`Config::fitness`; DEPRECATED 2026-11-03, A7 runs under `futhark_evolve` on the `futhark_sig` schemes; the shapes schemes stay registered, unchanged), phylu `examples/shape_fit.rs` (`ShapeConfig::fitness`) | `--fitness` |
| regex | phylu `examples/regex_fit.rs` (engine `regex_fitness_scheme`) | `EVOLVE_FITNESS`. Without it, `EVOLVE_REGEX_PARSIMONY=1` selects `regex-raw-length`, as it always did. If both are set and they disagree, the run stops and names the conflict. |

## Past experiments: commit and fitness as it was then

| experiment | commit | fitness then |
|---|---|---|
| futhark_evolve pathfinder R pilot, 2026-10-27 (not kept) | fuller 244cb948 | three log-scaled exactness measures a case, stage 2 ungated (no scheme) |
| The benchmark study, 2026-10-27/28 (`results/2026-10-27_31/fbench`: reduce, scan, histogram, pathfinder; R, C and P) | fuller 38216485 to 3b470301 (e.g. pathfinder_R_s1 at e64bd1f3, the P runs at 3b470301) | `futhark-study7` |
| Reruns on the fixed fitness, 2026-10-29 (`frerun2/point`, `frerun2/improve/histogram_R_s1`) | fuller 72c37606, f5be0978 | `futhark-slope4` |
| Gated reruns and seed-less runs, 2026-10-29 09:31Z onward (`frerun2/improve/*_gate`, `*_noseed*`) | fuller f465a5cf, 1aec1632, 514602a3 | `futhark-slope4-gated` |
| futhark_pixel_evolve, 2026-10-27 16:35 to 18:56 | fuller 8834c828 | TrueNorth(θ_q, θ_s), no stages (no scheme) |
| futhark_pixel_evolve, 2026-10-27 18:56 to 2026-10-29 10:30 | fuller 922ec1a6 | `staged` |
| wgsl_true_case_fit, 2026-10-28 11:07 to 2026-10-29 10:30 | phylu 528460b5 onward | `staged` |
| wgsl_true_case_fit k-means P runs, 2026-10-30/31 (`fbench/kmeans_P_s1`, the champion) | phylu at those dates (see each run's log header) | `staged-gated` |
| Shape kingdom A7 runs and the GECCO poster, 2026-10-27 (`docs/measurements/2026-10-27_futhark_shapes_A7*.md`) | fuller f3972766, 2911426c | `shapes-two-band` |
| Shape runs, 2026-10-27 17:22 to 2026-11-01 01:59 (none recorded) | fuller b1bccf66 | `shapes-hff` |
| Regex postcode runs, 2026-10-04 (phylu `logs/REGEX_POSTCODE`, `_MT`) | phylu 1d0d8ffc | generation-0 scaling with the length objective on (no scheme; `regex-colmax` is the same without length) |
| Regex runs, 2026-10-05 to 2026-11-01 02:02Z | phylu 8b05aaa5 onward | `regex-colmax` |

## Other in-place changes found in the git log (not schemes)

- **fuller `futhark_evolve`, speed inputs.**
  - f1c8bd17: the default went from 3 pairs to 11.
  - b1fe8606: a program that does no GPU work now gets speed input 1.
  - 349fc808: micro benchmarks are timed on n = 2^24.
  - 514602a3: the Futhark work bound replaced the WGSL cap.
- **phylu `wgsl_true_case_fit`, speed group.**
  - 30f2339a and 344fc0bd: the cap and the 2× rules.
  - d70991d2 and ec98a415: the timing case's wrong share joins the quality group.
  - The timing protocol: 63a4d353, 47db2f14, 404b6ab0, e794b799, bc613c2f.
- **phylu regex objective set.**
  - de177358: the 8-column confusion matrix became the default.
  - 4809a641: `EVOLVE_REGEX_FP_WEIGHT` defaults to 2.
  - 8b05aaa5: `FPR_ONLY` changed meaning.
- **phylu `image_fitness` / `ingest_fitness`.** 40d5f892, 05ff7676, d132cc1e, 80c02d2b.
- **phylu SR score kernels.** 65be165a and cb5d9917, numerical.
