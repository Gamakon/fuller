# The standard every evolution runner meets

**Objective.** fuller and phylu are one system, split only so each half can be supported on its own. Every runner that
evolves anything, in either repo, meets the same standard, so that Andrew can watch, stop, resume and audit any run
the same way.

**The standard.**

1. **Telemetry stream**: `OUT/stream.jsonl`, written through `fuller::telemetry` (`RunTelemetry`), the one format the
   watchers read (`phylu-sr-watch --follow OUT`, `scripts/phylu_web_watch.py --stream OUT/stream.jsonl`).
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

| piece | home | status (2026-10-27) |
|---|---|---|
| telemetry stream | `fuller::telemetry` | moving now |
| two-level HFF | hff library; the grouping is phylu `evolve::image_fitness::HffGroups` | to move to fuller |
| checkpoints | phylu `evolve::checkpoint` (engine), `true_wgsl_fit` own | to unify |
| kill trigger | phylu `evolve::engine::{RUN_FILE_ENV, run_file_present}` | to move to fuller |
| guards, hall of fame, ledger | each runner its own | to unify |

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
