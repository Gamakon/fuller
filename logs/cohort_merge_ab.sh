#!/bin/zsh
# THE COHORT-MERGE A/B.
#
# The run that scored 75 of 133 set cohort_merge = 10,000 against fits of a few
# thousand generations. Under either banding rule the elder band was never
# reached, so the cohorts stayed permanently separate for the whole run -- and
# the paper's 4.3 describes a merge that never fired. This asks whether the
# result DEPENDS on that.
#
# Three arms, everything else at the 75 settings: 800 intake + 400 champion,
# pump_every 100, gene_subsets on, max_generations 50000 so TIME binds, 360 s
# a fit, six objectives (no third block, as the 75 run).
#
#   cohort_merge 10000  never merges, as the 75 run
#   cohort_merge   500  merges at 5 pump beats
#   cohort_merge   100  merges at 1 pump beat -- the engine's own test calls a
#                       merge inside one beat "ALPS switched off wearing
#                       labels", so this is the near-off control, kept at the
#                       value asked for rather than silently moved to 200.
#
# THE WHOLE SCRIPT IS ONE FUNCTION so an edit mid-run cannot splice itself into
# a running shell (logs/bacres1_test.sh records what that cost).
set -u

run_arm() {

ROOT=/Users/andrewmorgan/Dev/gamakon/fuller/.claude/worktrees/agent-a53308d5df65d810c
DATA=$1
DS=$2
SEED=$3
MERGE=$4

TAG=cm${MERGE}_${DS}_s${SEED}
OUT=$ROOT/logs/cohort_ab/$TAG
mkdir -p $OUT

# The 75-of-133 settings, verbatim from docs/EXPERIMENTS.md:513-516 and the
# running-fits skill.
export EVOLVE_POP_INTAKE=800
export EVOLVE_POP_CHAMPION=400
export EVOLVE_PUMP_EVERY=100
export EVOLVE_GENE_SUBSETS=1
export EVOLVE_MAX_GENERATIONS=50000
export EVOLVE_SECONDS=${SECS:-360}
export EVOLVE_SEED=$SEED

# THE VARIABLE. One knob drives both islands: champion_cohort_merge defaults to
# None and falls through to config.cohort_merge (engine.rs:2884), and
# champion_open_fight is false at HEAD (engine.rs:853), so VIRTUAL ALPS runs on
# the intake AND the champion island under this single value.
export EVOLVE_COHORT_MERGE=$MERGE

# No third block -- the 75 run had six objectives.
export EVOLVE_SMOGD=0
export EVOLVE_SMOTE=0

# Its OWN telemetry and card. Two fits sharing either means the second
# destroys the first's record.
export EVOLVE_TELEMETRY_FILE=$OUT/stream.jsonl
export EVOLVE_TELEMETRY_RUN_ID=$TAG
export EVOLVE_CARD_OUT=$OUT/card.json
export EVOLVE_PROGRESS_EVERY=200

mkdir -p $ROOT/logs/cards
ln -sfn "$OUT/card.json" "$ROOT/logs/cards/$TAG.json"

echo "=== $TAG ==="
echo "data         $DATA"
echo "cohort_merge $MERGE   (pump_every 100 -> $(( MERGE / 100 )) live bands)"
echo "seed         $SEED (development; SRBench's official seeds are the test set)"
echo "budget       360 s, 50000 generation cap so TIME binds"
echo "telemetry    $EVOLVE_TELEMETRY_FILE"
echo "card         $EVOLVE_CARD_OUT"
echo

cd $ROOT
nice -n 10 ./target/release/examples/evolve_fit "$DATA"
echo "RUN DONE $TAG"
}

run_arm "$@"
