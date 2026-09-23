#!/bin/zsh
# THE A/B FOR TYPED TRANSCENDENTAL DEPTH -- docs/SPEC_typed_transcendental_depth.md.
#
# The spec's claims are all about the distribution of SRBench's TRUE LAWS: zero
# directly nested transcendental pairs, no law past depth 2. None of that is a
# claim about SEARCH BEHAVIOUR, and the two are different things. The risk the
# spec names and cannot answer is that the search needs illegal shapes as
# STEPPING STONES -- a blob at depth 4 as the waypoint to a depth-1 law. Only
# this measurement answers it.
#
# Two arms, same seed, same everything else: typed_depth unset against Some(2).
# Three seeds. The settings are the skill file's -- 800 + 400, pump 100,
# cohort_merge 10000, gene_subsets on -- which is the configuration that has
# actually recovered laws, not an invented one.
set -u

# THE WHOLE SCRIPT IS ONE FUNCTION, CALLED AT THE END. zsh reads a script
# incrementally, so an edit mid-run re-reads at the old byte offset and has
# before now run a fit a second time, truncating the first's telemetry. A
# function body is parsed whole before it can be called.
run_ab() {

ROOT=/Users/andrewmorgan/Dev/gamakon/fuller/.claude/worktrees/agent-a5ccf50f291890f3a
# The dataset lives beside the run, in the worktree, so the A/B does not depend
# on a session scratchpad that another run may not have.
DATA=${DATA_OVERRIDE:-$ROOT/logs/typed_ab/bacres1.tsv}

# THE SETTINGS THAT HAVE RECOVERED LAWS (.claude/skills/running-fits/SKILL.md).
# Not invented here: 800 + 400 runs 27 ms/generation, which is what makes a
# short A/B a real one rather than a handful of beats.
export EVOLVE_POP_INTAKE=800
export EVOLVE_POP_CHAMPION=400
export EVOLVE_PUMP_EVERY=100
export EVOLVE_COHORT_MERGE=10000
export EVOLVE_GENE_SUBSETS=1
# TIME BINDS, not generations: both arms get the same wall clock, so a
# difference in ms/generation shows up as generations rather than as seconds.
export EVOLVE_MAX_GENERATIONS=1000000
export EVOLVE_SECONDS=${SECS:-420}
# No SMOGD/SMOTE -> 6 objectives, as the 75-of-133 configuration ran.
export EVOLVE_SMOGD=0
export EVOLVE_SMOTE=0
export EVOLVE_PROGRESS_EVERY=200

for SEED in 7014 7015 7016; do
  for ARM in off on; do
    TAG=typed_${ARM}_${SEED}
    OUT=$ROOT/logs/$TAG
    mkdir -p $OUT $ROOT/logs/cards
    if [[ $ARM == on ]]; then
      export EVOLVE_TYPED_DEPTH=2
    else
      unset EVOLVE_TYPED_DEPTH
    fi
    export EVOLVE_SEED=$SEED
    export EVOLVE_TELEMETRY_FILE=$OUT/stream.jsonl
    export EVOLVE_TELEMETRY_RUN_ID=$TAG
    export EVOLVE_CARD_OUT=$OUT/card.json
    ln -sfn "$OUT/card.json" "$ROOT/logs/cards/$TAG.json"
    ln -sfn "$OUT/stream.jsonl" "$ROOT/logs/latest.jsonl"
    echo "=== $TAG : typed_depth=${EVOLVE_TYPED_DEPTH:-unset} seed=$SEED ${EVOLVE_SECONDS}s ==="
    cd $ROOT
    # A SNAPSHOT of the binary, not the build tree's: a rebuild while the A/B is
    # in flight would otherwise swap the engine under a running arm and the two
    # arms would no longer differ by one setting.
    $ROOT/logs/typed_ab/evolve_fit_a "$DATA" 2>&1 | tee $OUT/run.log
    echo "--- $TAG done ---"
  done
done
echo "AB DONE"
}

run_ab "$@"
