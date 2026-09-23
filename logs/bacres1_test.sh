#!/bin/zsh
# THE TEST RUN before the big one: strogatz_bacres1, 2,000 + 2,000 for 2,000
# generations. The shape we are aiming at is 200,000 + 200,000 for 20,000
# generations -- what a genetic algorithm that size can do on a GPU, which
# nobody has measured because nobody has been able to run it. This is the
# rehearsal: same engine, same switches, a hundredth of the population and a
# tenth of the generations.
#
# Everything the engine decides for itself is now a default (pump 33, ALPS at
# five cohorts, 3% promotion, snap, the beam, HFF on the device). What is set
# here is what only a RUN can know: the data, the population, the budget, the
# synthetic third block, and where the checkpoint and the telemetry go.
set -u

# THE WHOLE SCRIPT IS ONE FUNCTION, CALLED AT THE END.
#
# zsh reads a script INCREMENTALLY, so editing this file while a run is in
# flight makes the running shell re-read it at its old byte offset -- which
# lands mid-word and then executes whatever follows. That happened: an edit
# during the 20,000-row run printed "command not found: ry", fell into the new
# lines, and RAN THE FIT A SECOND TIME. The second run resumed from the
# checkpoint, was already finished, and its run_start TRUNCATED the telemetry
# stream the first run had written -- 678 records gone.
#
# A function body must be parsed whole before it can be called, so an edit
# mid-run can no longer splice itself into a running script.
run_fit() {

ROOT=/Users/andrewmorgan/Dev/gamakon/fuller
DATA=${1:-/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/b3c1f1fe-1035-445f-ada0-9f05e732d5e6/scratchpad/bacres1.tsv}
TAG=${2:-bacres1_test}
OUT=$ROOT/logs/$TAG

mkdir -p $OUT

# POPULATION and BUDGET, the two things this run is actually asking about.
# Defaults are the rehearsal's; the big shape is 200,000 + 200,000 for 20,000
# generations and the way there is to raise these.
#
# The seconds cap is high enough that GENERATIONS end the run: the question is
# what the search does with N beats, not what it does in a fixed minute. A fit
# that meets the stop bar ends earlier than either.
export EVOLVE_POP_INTAKE=${POP:-2000}
export EVOLVE_POP_CHAMPION=${POP:-2000}
export EVOLVE_MAX_GENERATIONS=${GENS:-2000}
export EVOLVE_SECONDS=${SECS:-86400}

# THE DEVELOPMENT SEED. 7014 is ours; SRBench's official seeds are the test set
# and are only used for a final run.
export EVOLVE_SEED=7014

# SMOGD AND SMOTE -- the synthetic third block. These cannot be engine defaults:
# the rows are GENERATED in evolve_fit from the data it was handed (a 2D UMAP of
# the inputs, an adaptive grid, new rows drawn from their nearest real rows), so
# they only exist once a dataset does. They are noisy on purpose: their errors
# rank individuals in the tournaments and never decide that a fit is exact.
export EVOLVE_SMOGD=1
export EVOLVE_SMOTE=1

# THE CHECKPOINT. Five rotating slots, one a minute. This is what makes the run
# stoppable: a fit killed at any moment resumes from the beat before and is
# indistinguishable from one that ran straight through.
export EVOLVE_CHECKPOINT_DIR=$OUT/checkpoint
export EVOLVE_CHECKPOINT_EVERY=60

# THE TELEMETRY STREAM, which hff-watch repaints from.
export EVOLVE_TELEMETRY_FILE=$OUT/stream.jsonl
export EVOLVE_TELEMETRY_RUN_ID=$TAG
export EVOLVE_PROGRESS_EVERY=20

echo "=== $TAG ==="
echo "data       $DATA"
echo "population $EVOLVE_POP_INTAKE intake + $EVOLVE_POP_CHAMPION champion"
echo "budget     $EVOLVE_MAX_GENERATIONS generations, $EVOLVE_SECONDS s cap"
echo "seed       $EVOLVE_SEED (development, not an SRBench seed)"
echo "block 3    SMOGD + SMOTE"
echo "checkpoint $EVOLVE_CHECKPOINT_DIR every ${EVOLVE_CHECKPOINT_EVERY}s"
echo "telemetry  $EVOLVE_TELEMETRY_FILE"
# NOT a hardcoded list of the engine's settings -- one said "pump 33" while the
# binary ran 20, which is the kind of line that gets believed. The run_start
# record in the telemetry stream is what the engine actually used; read that.
echo "engine     defaults from the binary -- see run_start in the telemetry stream"
echo
# THE MONITOR COMMAND, printed BEFORE the fit starts and not after. Every run
# writes its own stream, so the path changes every time and a path given after
# the launch is a path read while watching the previous run.
ln -sfn "$OUT/stream.jsonl" "$ROOT/logs/latest.jsonl"
echo "WATCH THIS:"
echo "  cd $ROOT && ./target/release/hff-watch --follow logs/latest.jsonl"
echo "  (logs/latest.jsonl always points at the newest run; this one is $TAG)"
echo

cd $ROOT
./target/release/examples/evolve_fit "$DATA"
echo "RUN DONE"
}

run_fit "$@"
