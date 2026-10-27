# The standard every evolution runner meets

**Objective.** fuller and phylu are one system, split only so each half can be supported on its own. Every runner that
evolves anything, in either repo, meets the same standard, so that Andrew can watch, stop, resume and audit any run
the same way.

**The standard.**

1. **Telemetry stream**: `OUT/stream.jsonl`, written through `fuller::telemetry` (`RunTelemetry`), the one format the
   watcher reads (`evowatch OUT` in a terminal, `evowatch OUT --web PORT` in a browser; phylu `docs/WATCHER.md`).
2. **Two-level HFF** from the hff library: one TrueNorth angle over the quality inputs, one over the speed inputs,
   then HFF of the two angles. A runner with no speed inputs carries one group.
3. **Checkpoints and resume.**
4. **The kill trigger**: `OUT/killme.trigger` written at the start; deleting it ends the run cleanly after the current
   generation (never a signal).
5. **Guards**: a wall-clock cap and a work bound, so a runaway search cannot hold the machine.
6. **Hall of fame with pictures and text**: the best individuals, with their program text and, where the kingdom
   draws, a picture.
7. **Ledger and lineage**: one line per scored individual, with its parents.

**Home of each shared piece.** fuller, because phylu depends on fuller and fuller never on phylu. phylu re-uses it and
keeps no second copy.

| piece | home | status (2026-10-27, evening) |
|---|---|---|
| telemetry stream | `fuller::telemetry` | moved; phylu re-exports it |
| two-level HFF | hff library's TrueNorth; the grouping is phylu `evolve::image_fitness::HffGroups` (WGSL), `futhark_shapes_evolve` and `shape_typed::two_level` (shapes), `futhark_pixel` (pixels) | every runner with speed inputs is two-level; the grouping code is not yet one piece |
| checkpoints | phylu `evolve::checkpoint` (engine), `true_wgsl_fit`, the shape loops, `futhark_pixel_evolve` each their own | every runnable runner writes them by default and resumes only on `--resume` |
| kill trigger | `fuller::run_file`; phylu re-exports it | moved |
| guards, hall of fame, ledger | each runner its own | every runnable runner has them |

## Inventory, 2026-10-27 (before the work)

Y = meets it; opt-in = only when an environment variable or flag names a path; N = does not.

| runner | repo | loop | stream | two-level HFF | checkpoint + resume | kill trigger | guards | hall of fame | ledger / lineage |
|---|---|---|---|---|---|---|---|---|---|
| `examples/evolve_fit` (SR) | phylu | `evolve::engine` | opt-in (`EVOLVE_TELEMETRY_FILE`) | N (one flat HFF) | opt-in (`EVOLVE_CHECKPOINT_DIR`) | opt-in (`EVOLVE_RUN_FILE`) | Y (time, 1M evaluations) | text, no picture | opt-in genealogy |
| `examples/regex_fit` | phylu | `evolve::engine` | opt-in | N | N (engine can; runner does not wire it) | opt-in | Y | text (rule export) | N |
| `examples/kernel_time` | phylu | `evolve::engine` | N (off on purpose: a timing harness) | N | N | N | Y | N | N |
| `examples/wgsl_fit` (legacy) | phylu | `evolve::wgsl_typed_fit` | N | N | N | Y | Y (seconds) | `best.wgsl` only | Y (`typed_ledger.tsv`) |
| `examples/wgsl_tune` | phylu | drives `wgsl_fit` | as `wgsl_fit` | as `wgsl_fit` | N | as `wgsl_fit` | Y | `tuned.wgsl` | as `wgsl_fit` |
| `examples/wgsl_true_fit` | phylu | `evolve::true_wgsl_fit` | Y (with a checkpoint dir) | N | Y | N | Y | text | opt-in (`lineage.tsv`, marriage on) |
| `examples/wgsl_true_image_fit` (volumepassage) | phylu | `evolve::true_wgsl_fit` | Y | Y | Y | N | Y | text + pictures | opt-in |
| `examples/wgsl_true_ingest_fit` | phylu | `evolve::true_wgsl_fit` | Y | Y | Y | Y (own check) | Y | text | opt-in |
| `examples/shape_fit` | phylu | `evolve::shape_typed` | N | N | N | N | Y (seconds) | N (best only) | N |
| `examples/futhark_shapes_evolve` | fuller | `futhark_shapes_evolve::evolve` | N | N | N | N | N (generations only) | N (best only) | N |
| `examples/wgsl_gram_evolve` | fuller | its own | N | N | N | N | Y (build watchdog) | N | Y (`ledger.tsv`) |
| `examples/futhark_pixel_evolve` | fuller | `futhark_pixel` (another agent) | N | Y | N | N | N | text + pictures | N (series only) |

## Inventory, 2026-10-27 (after the work)

Every runner meets every item, or the cell gives the written reason it does not.

| runner | stream | two-level HFF | checkpoint + resume | kill trigger | guards | hall of fame | ledger / lineage | mutation events |
|---|---|---|---|---|---|---|---|---|
| fuller `futhark_shapes_evolve` | Y | Y (quality: disagreement per input set; speed: traffic, allocations, depth) | Y (`--resume`) | Y | Y (`--seconds`, generations) | text | Y (`ledger.tsv`, parents) | `egraph_exact`, `write_back_defect` |
| fuller `futhark_pixel_evolve` | Y | Y (θ_quality: the picture; θ_speed: the timed pairs) | Y (`OUT/checkpoint.json`, `--resume`) | Y | Y (`--minutes`, the work bound) | text + pictures + OCR | Y (`ledger.tsv`, parents) | `egraph_exact`, `saturated_fold`, `pump` |
| fuller `futhark_evolve` (the study runner, `docs/STUDY_evolving_benchmarks.md`) | Y | Y, staged: θ_quality alone (seven linear inputs a case) until the best is exact on every case, then θ_quality and θ_speed, the speed over the program text's 11 timing slots (`runner::TimingSlots`, each pair its own input); `--stop-at-exact` (R, C) ends before stage 2 | Y (`OUT/checkpoint.json`, `--resume`) | Y | Y (`--minutes`, `--evaluations`, each candidate's node executions against the per-submit cap before it runs) | text (`OUT/hall_of_fame/`, top 5) + the HFF chart | Y (`ledger.tsv`, parents) | `egraph_exact`, `pump`, `saturated_fold`, `marriage`, `stage_change` |
| phylu `shape_fit` (`shape_typed`) | Y | Y (as `futhark_shapes_evolve`) | Y (`--resume`) | Y | Y | text | Y (the device's parent buffer) | `egraph_exact`, `write_back_defect` |
| phylu `wgsl_true_fit` (the Gram problem) | Y | Y (`--hff-groups quality,speed`, the default since 2026-10-27: θ_quality over the three tolerance ratios, θ_speed over `r/(1+r)` of the device time against the shipped kernel; `flat` is the old one TrueNorth) | Y (`--resume`) | Y | Y | text | Y (`lineage.tsv`) | every optimiser, `adf_fold`, `marriage`, `pump`, `write_back_defect` |
| phylu `wgsl_true_image_fit`, `wgsl_true_ingest_fit` | Y | Y | Y (`--resume`) | Y | Y | text (+ pictures, image) | Y (`lineage.tsv`) | as `wgsl_true_fit` |
| phylu `evolve_fit` (SR), `regex_fit` (engine) | Y (`EVOLVE_OUT`) | NOT APPLICABLE: one quality group. Andrew's rule, "Quality first; HFF selects on quality only": the engine has no speed input, and speed is a second phase on settled models, never an HFF column | Y: `OUT/checkpoint/` by default (every 60 s, `EVOLVE_CHECKPOINT_EVERY`); resumed ONLY with `--resume` (an error when there is nothing to resume); a fresh run moves an earlier run's slots to `checkpoint/superseded/<UTC>/` | Y | Y (seconds, 1M evaluations) | text | Y: the genealogy by default (`OUT/genealogy.tsv`; `EVOLVE_GENEALOGY=0` turns it off), measured below | the engine's `snap`, `fold`, `typed_snap`, `regex_simplify`, `pump` |
| phylu `kernel_time` | EXEMPT: a timing harness, not a search; every stop and the stream are off on purpose so the kernel's time is measured alone | | | | | | | |
| fuller `wgsl_gram_evolve`, phylu `wgsl_fit`, phylu `wgsl_tune` | LEGACY: runners of the old WGSL kingdom, which has no work bound; fuller's work guard refuses every device dispatch of it (`Fuel::Unported`), so they cannot run. `wgsl_tune` drives the `wgsl_fit` binary. The Gram problem runs on the true kingdom (`wgsl_true_fit`) | | | | | | | |

The standard's tests name every exclusion: fuller `tests/runner_standard.rs` (`LEGACY`) and phylu
`tests/runner_standard.rs` (`LEGACY`, `EXEMPT`), each with its reason and the line in the runner's source that stops
it, so the day that line goes the test fails until the runner is brought to the standard. phylu's registry also
records each runner's HFF: two-level (with the source marker that selects it) or one quality group with its reason.

**The genealogy's cost (2026-10-27, measured; the gate was 5 % of wall time).** Full population (2000 + 2000), M-series
laptop, genealogy on and off alternated, the same seed; the search is bit for bit the same either way (same model,
same fitness, same counts).

| run | measurement | genealogy on | genealogy off | the genealogy's own timer |
|---|---|---|---|---|
| SR, `feynman_III_9_52`, 200 generations, 6 pairs | ms per generation | 106.6, 107.0, 110.2, 113.1, 129.7, 132.6 | 106.5, 113.0, 108.1, 128.3, 119.6, 128.8 | 0.20 to 0.39 s of 21 to 27 s: 1.0 to 1.5 % |
| regex, `uk_postcode`, 60 generations, 4 pairs | seconds | 70.7, 59.3, 107.7, 75.3 | 72.7, 58.2, 82.0, 84.4 | 0.44 to 0.79 s: 0.6 to 1.0 % |

The on/off wall times differ by less than the machine's run-to-run spread (other work shared it: the first two pairs
of each ran beside it), and the genealogy's own timer puts it at 1.5 % at most, under the 5 % gate, so it is on by
default.

LEAF is a mutation mode of the device's own variation; the device does not report which rows it made, so there is no
`leaf` event yet (its share is in every WGSL snapshot). The tests that fail when a runner leaves the standard:
fuller `tests/runner_standard.rs`, phylu `tests/runner_standard.rs` and phylu `shape_typed`'s
`the_shape_run_meets_the_standard` (GPU).

Not runners (no generation loop over a population, checked): phylu `wgsl_typed_probe`, `init_size_study`,
`order_study`, `population_dag_ab`, `fold_to_dag_measure`, `st8b_leaf_measure`, `st13_rescore`,
`regex_holdout_score`, `kitchen_sink`; fuller `wgsl_gram_hgp` (one round of model calls), `population_*`,
`st9_champion_time`, `wgsl_mine_rewrites`, `wgsl_symmetric_table`.

## The telemetry stream (`fuller::telemetry`)

One JSON object a line in `OUT/stream.jsonl`, tagged by `type`. Every record carries the header `schema_version`
(1), `run_id`, `seq` (gapless), `timestamp_utc`, `elapsed_ms`, `generation`. The record types:

| `type` | Rust type | what |
|---|---|---|
| `run_start` | `RunStart` | the run's fixed facts: `kingdom`, `dataset`, `seed`, `population`, `max_generations`, `budget_ms`, `hff_objectives`, `config_json`, the SR/regex/WGSL blocks |
| `snapshot` | `Snapshot` | one generation: `global` (`Global`: `best_hff`, `best_ever_hff`, `avg_hff`, `median_hff`, `nan_rows`, `angle_groups`, `extras`, diversity `genes`/`distinct_genes`, `evaluations`, the kingdom blocks), `islands`, `global_cohorts` |
| `event` | `Event` | something that happened, of an `EventKind`; mutation events below |
| `model` | `Model` | the best model's text (SR) |
| `hall_of_fame` | `HallOfFame` | the hall of fame, best first (`HofRow`: `rank`, `hff`, `born`, `tree`, `nodes`, `gpu_ms`, `picture`, `extras`) |
| `run_end` | `RunEnd` | `stopped_by` (`n_gen`, `time`, `killed`, `early_stop`, `error`), `generations`, `individuals`, `best_hff`, `seconds` |

A runner writes it through `RunTelemetry` (`start` → `generation` each generation → `hall_of_fame` / `mutation` →
`end`); a loop that unwinds without `end` writes `stopped_by: "error"`. `RunTelemetry::start` also writes
`OUT/killme.trigger`; `stop_requested()` is true once it is deleted. Every field added since the format moved here is
omitted when unset, so older streams are byte for byte what they were (phylu `tests/telemetry_byte_compat.rs`).

## Mutation events

Every mutation outside plain GEP variation is one `event` record, written by `RunTelemetry::mutation` (or
`Writer::mutation`). Fields beyond the header:

| field | meaning |
|---|---|
| `kind` | the `EventKind`: `fold`, `snap`, `typed_snap`, `cleanse`, `egraph_exact`, `egraph_real` (value-changing), `hoist`, `trips`, `adf_fold`, `saturated_fold`, `let_share`, `marriage`, `subtree_crossover`, `leaf`, `pump`, `model_rewrite` (an LLM's edit), `write_back_defect`, `regex_simplify` |
| `operator` | the operator's own name, finer than `kind` |
| `message` | one line for a human |
| `row`, `id`, `parents` | the individual's row, its id, its parents' ids (lineage) |
| `text_before`, `text_after` | the kingdom's own text before and after (subtree, formula, rule, kernel line), at most 240 characters each |
| `text_file` | `event_text/<seq>.txt` under `OUT`, holding both texts in full, when either was cut |
| `fitness_before`, `fitness_after` | the individual's HFF before and after, where known (`value` repeats `fitness_after`) |
| `entered`, `improved` | the child entered the population; it was better than its source |
| `nodes` | the size after, where it matters (folds) |
| `count` | a batched event stands for this many mutations (absent: one) |

For the ADF and saturated folds, `count` is the ADFs created, `nodes` the genes rewritten, and `text_after` the shared
part.
