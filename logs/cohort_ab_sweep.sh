#!/bin/zsh
# THE SWEEP: 3 arms x 3 seeds x 2 datasets = 18 fits at 360 s.
#
# The three ARMS of one seed-dataset run CONCURRENTLY, so all three see the
# same machine contention -- which is what matters when TIME binds the budget
# and a fit's generation count is the thing being compared. Rounds are
# sequential: 6 rounds x ~6 min.
set -u

sweep() {
ROOT=/Users/andrewmorgan/Dev/gamakon/fuller/.claude/worktrees/agent-a53308d5df65d810c
SP=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/b3c1f1fe-1035-445f-ada0-9f05e732d5e6/scratchpad/data

for DS in bacres1 barmag2; do
  for SEED in 7013 7014 7015; do
    echo "########## round $DS seed $SEED : arms 10000 / 500 / 100 concurrently"
    for MERGE in 10000 500 100; do
      zsh $ROOT/logs/cohort_merge_ab.sh $SP/$DS.tsv $DS $SEED $MERGE \
        > $ROOT/logs/cohort_ab/${DS}_s${SEED}_cm${MERGE}.log 2>&1 &
    done
    wait
    echo "########## round $DS seed $SEED DONE"
  done
done
echo "SWEEP DONE"
}

sweep
