# Experiments — how to run one, without hunting

Every knob, in one place. If you had to grep the source to find a setting, that
is a bug in this file — fix it here.

## Run one

```bash
cd /Users/andrewmorgan/Dev/gamakon/fuller

EVOLVE_SEED=7014 \
EVOLVE_POP_INTAKE=4000 EVOLVE_POP_CHAMPION=4000 \
EVOLVE_MAX_GENERATIONS=40000 EVOLVE_SECONDS=86400 \
EVOLVE_PUMP_EVERY=100 EVOLVE_COHORT_MERGE=10000 \
EVOLVE_GENE_SUBSETS=1 \
EVOLVE_TYPED_DEPTH=2 \
EVOLVE_TELEMETRY_FILE=$PWD/logs/<tag>/stream.jsonl \
EVOLVE_TELEMETRY_RUN_ID=<tag> \
EVOLVE_CHECKPOINT_DIR=$PWD/logs/<tag>/checkpoint \
EVOLVE_CHECKPOINT_EVERY=60 \
EVOLVE_PROGRESS_EVERY=500 \
nohup nice -n 5 ./target/release/examples/evolve_fit <data.tsv> \
  > logs/<tag>/run.log 2>&1 &
```

Then, always:

```bash
ln -sfn $PWD/logs/<tag>/stream.jsonl logs/latest.jsonl
./target/release/hff-watch --follow logs/latest.jsonl
```

`logs/latest.jsonl` never changes, so one monitor command works for every run.

## Every environment variable

Grouped by what it decides. **Unset means the engine's default**, which is in
`Config::srbench` in `src/evolve/engine.rs`.

### The shape of the run

| variable | what it does |
|---|---|
| `EVOLVE_SEED` | the seed. **Use DEV seeds (7013–7016). SRBench's official seeds are the test set.** |
| `EVOLVE_POP_INTAKE` | intake island size |
| `EVOLVE_POP_CHAMPION` | champion island size |
| `EVOLVE_MAX_GENERATIONS` | generation cap |
| `EVOLVE_SECONDS` | wall-clock cap. Set high if you want generations to bind. |
| `EVOLVE_PAIRS` | island pairs (default 1) |
| `EVOLVE_HEAD` | gene head length (default 34) |
| `EVOLVE_GENES` | genes per chromosome (default 3) |
| `EVOLVE_RNC_LO` / `_HI` | random constant range (default −100..100) |

### TSR — Transcendental Symbolic Regression

| variable | what it does |
|---|---|
| `EVOLVE_TYPED_DEPTH` | **the TSR ceiling.** `2` is the measured value. UNSET is the untyped engine. `0` is refused, not read as "off". |

See `kingdoms/tsr/README.md` and `docs/SPEC_typed_transcendental_depth.md`.

### The pump and the islands

| variable | what it does |
|---|---|
| `EVOLVE_PUMP_EVERY` | the pump's beat in generations |
| `EVOLVE_COHORT_MERGE` | VIRTUAL ALPS: the AGE at which cohorts merge into one elder band. **0 turns ALPS off.** |
| `EVOLVE_CHAMPION_ALPS=1` | cohorts on the champion island too (the default) |
| `EVOLVE_CHAMPION_TOURNAMENT` | the champion island's own tournament fraction |
| `EVOLVE_CHAMPION_ELITES` | the champion island's own elite count |
| `EVOLVE_CHAMPION_COHORT_MERGE` | the champion island's own merge age |
| `EVOLVE_ARRIVAL_CHILDREN` | children per promoted arrival (default 4) |
| `EVOLVE_CROSS_EVERY` / `EVOLVE_K_MIGRANTS` | the cross step (default off) |

**`promote_fraction` (3%) has no env var** — it is compiled in.

### The gene editors

| variable | what it does |
|---|---|
| `EVOLVE_SNAP_EVERY` / `_TOP_K` | constant → symbolic form (π, e, φ), written back |
| `EVOLVE_FOLD_EVERY` | subexpression → constant, written back. **ON by default at the pump's beat.** |
| `EVOLVE_BEAM_EVERY` / `_WIDTH` | the beam. ON by default. |
| `EVOLVE_BEAM_TREE=1` | the beam's tree neighbourhood (off by default) |
| `EVOLVE_BEAM_WRAPS` | the beam's functional wraps (on with the beam) |

### HFF and the stop bar

| variable | what it does |
|---|---|
| `EVOLVE_TOWER=1` | the tower objective joins HFF |
| `EVOLVE_REDUNDANCY=1` | leave-one-gene-out joins HFF |
| `EVOLVE_HFF_LOG` / `_TRAIN` / `_VAL` / `_BLOCK3` | log-scale a block |
| `EVOLVE_HFF_NO_VAL=1` | leave validation out of HFF |
| `EVOLVE_STOP_LOG10_P` | the p half of the bar (default −19; `inf` switches it off) |
| `EVOLVE_BALANCED_TOURNAMENTS=1` | breed on the balanced pole |
| `EVOLVE_GENE_SUBSETS=1` | score every non-empty gene subset (15 combinations, 45 candidates) |
| `EVOLVE_COMPOUNDS=1` | compound functions join the symbol table |
| `EVOLVE_HFF_ON_HOST=1` | HFF on the host, not the GPU. **For parity checks only.** |

### Data

| variable | what it does |
|---|---|
| `EVOLVE_SMOGD=1` | synthetic third block (UMAP + adaptive grid) |
| `EVOLVE_SMOTE=1` | SMOTE rows join the third block |
| `EVOLVE_SMOGD_NOISE` | multiplier on the neighbours' variance |
| `EVOLVE_EDGE=file.tsv` | held-out edge rows as the third block |

**SMOGD/SMOTE add three HFF objectives**, taking 6 to 9. They rank in
tournaments and are excluded from the 1−R² half of the stop bar — but NOT from
the p half. That asymmetry is a known open question.

### Output

| variable | what it does |
|---|---|
| `EVOLVE_TELEMETRY_FILE` | the JSONL stream `hff-watch` reads |
| `EVOLVE_TELEMETRY_RUN_ID` | what the stream calls the run |
| `EVOLVE_PROGRESS_EVERY` | progress line every N generations. **Telemetry needs this > 0.** |
| `EVOLVE_CHECKPOINT_DIR` / `_EVERY` | five rotating slots; exact stop/start |
| `EVOLVE_CARD_OUT` | an extra copy of the run card (one is always written beside the log) |
| `EVOLVE_CARD` | **run FROM a card.** Env still wins over it. |
| `EVOLVE_HOF_FILE` / `EVOLVE_GENEALOGY_FILE` | hall of fame, lineage log |

## Run cards

Every run writes `card.json` beside its log, **before the fit starts**, holding
the whole `Config`, the dataset, SMOGD/SMOTE state, the git commit, and the
derived `hff_objectives` count.

```bash
diff logs/<a>/card.json logs/<b>/card.json     # they serialise in declaration order
EVOLVE_CARD=logs/<tag>/card.json ./target/release/examples/evolve_fit
```

**Write a note into the card when a run ends** — what was tested, what
happened, whether it recovered the law. `Note` is in `src/evolve/card.rs`. A
card with no note is a configuration nobody can learn from.

## Where things live

```
experiments/<name>/     one experiment: what was asked, settings, results
kingdoms/<kingdom>/     a symbol table and its measurements
  tsr/                  TSR — the typed transcendental kingdom
  symbolic-regression/  the untyped op set
logs/<tag>/             run.log, stream.jsonl, card.json, checkpoint/
logs/cards/             the card library
logs/latest.jsonl       symlink to the newest run's stream
docs/audit/             paper-vs-code alignment findings
.claude/skills/running-fits/SKILL.md   what has been got wrong before
```

## Read before running

`.claude/skills/running-fits/SKILL.md`. It holds the settings that have
actually recovered laws and the traps that have each cost an afternoon —
the p-value's dimension-freedom, `min_hff 0.000000` not meaning success, and
why you must never edit a script while it is running.
