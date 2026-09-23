#!/bin/zsh
# THE THIRD ARM: init AND point mutation typed, against the first run's two.
#
# The first A/B typed only `init`. That measures the ceiling on the population
# a fit is BORN with, and point mutation then puts towers straight back:
# `SymbolTable::wide` has 18 functions and 10 of them raise depth, so 56% of a
# point draw's function choices raise one. A third arm on the same three seeds
# turns one unexplained refusal rate into an ATTRIBUTION -- how much of it is
# mutation, and how much is crossover and transposition moving legal spans into
# places that make them illegal.
#
# Same binary discipline as the first run: a snapshot, not the build tree's.
set -u

run_arm3() {

ROOT=/Users/andrewmorgan/Dev/gamakon/fuller/.claude/worktrees/agent-a5ccf50f291890f3a
DATA=${DATA_OVERRIDE:-$ROOT/logs/typed_ab/bacres1.tsv}

export EVOLVE_POP_INTAKE=800
export EVOLVE_POP_CHAMPION=400
export EVOLVE_PUMP_EVERY=100
export EVOLVE_COHORT_MERGE=10000
export EVOLVE_GENE_SUBSETS=1
export EVOLVE_MAX_GENERATIONS=1000000
export EVOLVE_SECONDS=${SECS:-420}
export EVOLVE_SMOGD=0
export EVOLVE_SMOTE=0
export EVOLVE_PROGRESS_EVERY=200
export EVOLVE_TYPED_DEPTH=2

for SEED in 7014 7015 7016; do
  TAG=typed_both_${SEED}
  OUT=$ROOT/logs/$TAG
  mkdir -p $OUT $ROOT/logs/cards
  export EVOLVE_SEED=$SEED
  export EVOLVE_TELEMETRY_FILE=$OUT/stream.jsonl
  export EVOLVE_TELEMETRY_RUN_ID=$TAG
  export EVOLVE_CARD_OUT=$OUT/card.json
  ln -sfn "$OUT/card.json" "$ROOT/logs/cards/$TAG.json"
  ln -sfn "$OUT/stream.jsonl" "$ROOT/logs/latest.jsonl"
  echo "=== $TAG : init+mut typed, ceiling 2, seed=$SEED ${EVOLVE_SECONDS}s ==="
  cd $ROOT
  $ROOT/logs/typed_ab/evolve_fit_b "$DATA" 2>&1 | tee $OUT/run.log
  echo "--- $TAG done ---"
done
echo "ARM3 DONE"
}

run_arm3 "$@"
