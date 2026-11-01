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
| **`futhark-slope4-gated`** | futhark_sig | fuller f465a5cf | Quality, per case: the wrong share; Spearman and Pearson disagreement; the shares off by over 50 % and 1 % (`QUALITY_PARTS`). Stage 2 is gated: an inexact program scores π/2 + θ_quality. | default |
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

## Runners

| family | runners | how to choose |
|---|---|---|
| futhark_sig | fuller `examples/futhark_evolve.rs` | `--fitness` |
| staged | phylu `examples/wgsl_true_case_fit.rs`, fuller `examples/futhark_pixel_evolve.rs` | `--fitness` |
| shapes | fuller `examples/futhark_shapes_evolve.rs` (`Config::fitness`), phylu `examples/shape_fit.rs` (`ShapeConfig::fitness`) | `--fitness` |
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
