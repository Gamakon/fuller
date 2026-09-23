#!/bin/zsh
# THE FOLD OPERATOR'S EVIDENCE RUN: a short strogatz_bacres1 fit that shows the
# fold firing DURING the search (mid-fit Fold events carrying the row they landed
# in) and the leave-one-out's drops arriving after it ends. Small on purpose --
# this is a recording for the viewer's fixture and the panel, not a search.
#
# THE WHOLE SCRIPT IS ONE FUNCTION, called at the end: zsh reads a script
# incrementally, and an edit mid-run splices itself into the running shell.
set -u
run_fit() {
ROOT=/Users/andrewmorgan/Dev/gamakon/fuller/.claude/worktrees/agent-ad429133994c0dd3b
DATA=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/b3c1f1fe-1035-445f-ada0-9f05e732d5e6/scratchpad/bacres1.tsv
OUT=$ROOT/logs/fold_operator
mkdir -p $OUT
export EVOLVE_POP_INTAKE=600
export EVOLVE_POP_CHAMPION=600
export EVOLVE_MAX_GENERATIONS=400
export EVOLVE_SECONDS=600
export EVOLVE_SEED=7014
export EVOLVE_SMOGD=1
export EVOLVE_SMOTE=1
export EVOLVE_TELEMETRY_FILE=$OUT/stream.jsonl
export EVOLVE_TELEMETRY_RUN_ID=fold_operator
export EVOLVE_PROGRESS_EVERY=20
cd $ROOT
cargo run --release --features gpu --example evolve_fit -- $DATA
}
run_fit
