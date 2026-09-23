#!/bin/zsh
# TSR ONLY -- the treatment, across problems and seeds.
#
# Andrew: "your first instinct is to do an A-B test, right? We don't need an A
# test. We've been running A all week, and it's failed every time. What I need
# is a B test. I want you to just only test TSR."
#
# So there is no control arm here. The untyped engine's results at these
# settings are on the record (docs/EXPERIMENTS.md, logs/cards/) and are cited
# as PRIOR RESULTS, never re-run.
#
# The budget goes on BREADTH instead: eight datasets whose true laws span the
# depths TSR types. Several are the cases the T2 rung exists for --
# feynman_I_26_2 is arcsin(n sin theta), depth 2 through a transcendental over
# a transcendental's argument, and if the ceiling is wrong anywhere it is
# there.
set -u

# THE WHOLE SCRIPT IS ONE FUNCTION, CALLED AT THE END. zsh reads a script
# incrementally, so an edit mid-run re-reads at the old byte offset -- which
# once ran a fit a second time and truncated the first's telemetry.
run_sweep() {

ROOT=/Users/andrewmorgan/Dev/gamakon/fuller/.claude/worktrees/agent-a5ccf50f291890f3a
BIN=$ROOT/logs/tsr/evolve_fit_tsr

# The settings that recovered 75 of 133 (.claude/skills/running-fits/SKILL.md).
export EVOLVE_POP_INTAKE=800
export EVOLVE_POP_CHAMPION=400
export EVOLVE_PUMP_EVERY=100
export EVOLVE_COHORT_MERGE=10000
export EVOLVE_GENE_SUBSETS=1
export EVOLVE_MAX_GENERATIONS=1000000
export EVOLVE_SECONDS=${SECS:-180}
export EVOLVE_SMOGD=0
export EVOLVE_SMOTE=0
export EVOLVE_PROGRESS_EVERY=500

# THE KINGDOM: tsr. The ceiling is 2 because that is where the true-model
# distribution puts it -- 3 laws at depth 2, none beyond.
export EVOLVE_TYPED_DEPTH=2

# DEVELOPMENT SEEDS. SRBench's official seeds are the test set and are used
# only for a final run.
for SEED in 7014 7015; do
  for DATA in $ROOT/logs/tsr/data/*.tsv; do
    NAME=$(basename $DATA .tsv)
    TAG=tsr_${NAME}_${SEED}
    OUT=$ROOT/logs/tsr/$TAG
    mkdir -p $OUT $ROOT/logs/cards
    export EVOLVE_SEED=$SEED
    export EVOLVE_TELEMETRY_FILE=$OUT/stream.jsonl
    export EVOLVE_TELEMETRY_RUN_ID=$TAG
    export EVOLVE_CARD_OUT=$OUT/card.json
    ln -sfn "$OUT/card.json" "$ROOT/logs/cards/$TAG.json"
    ln -sfn "$OUT/stream.jsonl" "$ROOT/logs/latest.jsonl"
    echo "=== $TAG : kingdom=tsr ceiling=2 seed=$SEED ${EVOLVE_SECONDS}s ==="
    cd $ROOT
    $BIN "$DATA" 2>&1 | tee $OUT/run.log
    echo "--- $TAG done ---"
  done
done
echo "TSR SWEEP DONE"
}

run_sweep "$@"
