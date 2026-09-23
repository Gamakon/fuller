#!/bin/zsh
# THE FOLD OPERATOR A/B: the same fit with the operator on and off, same seed,
# same shape, so the difference is the operator and nothing else. It answers the
# only question the cost number cannot: does buying back head space pay for the
# beat it costs?
set -u
run_ab() {
ROOT=/Users/andrewmorgan/Dev/gamakon/fuller/.claude/worktrees/agent-ad429133994c0dd3b
DATA=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/b3c1f1fe-1035-445f-ada0-9f05e732d5e6/scratchpad/bacres1.tsv
OUT=$ROOT/logs/fold_operator
mkdir -p $OUT
export EVOLVE_POP_INTAKE=600
export EVOLVE_POP_CHAMPION=600
export EVOLVE_MAX_GENERATIONS=400
export EVOLVE_SECONDS=600
export EVOLVE_SMOGD=1
export EVOLVE_SMOTE=1
export EVOLVE_PROGRESS_EVERY=100
unset EVOLVE_TELEMETRY_FILE
cd $ROOT
# Three seeds, because one fit is an anecdote.
for SEED in 7014 7015 7016; do
  export EVOLVE_SEED=$SEED
  for MODE in on off; do
    if [[ $MODE == off ]]; then export EVOLVE_FOLD_EVERY=0; else unset EVOLVE_FOLD_EVERY; fi
    echo "=== seed $SEED fold=$MODE ==="
    cargo run --release --features gpu --example evolve_fit -- $DATA 2>/dev/null | grep -E "^HFF|^GENERATIONS|^FOLD\tbeats|R. on the unseen|ms per generation"
  done
done
}
run_ab
