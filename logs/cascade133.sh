#!/bin/zsh
# THE 133, ordered easy-first by the generation each law was solved at.
# Three stages, time-binding: 60 s, then 120 s on survivors, then 180 s.
# TSR on, 2000+2000, at the settings that recovered 75 of 133.
#
# ONE LOG PATH FOR THE WHOLE CASCADE. Never changes, never a symlink:
#
#   logs/RUN/stream.jsonl    every law, appended, for hff-watch --follow
#   logs/RUN/sweep.log       one line per law as it finishes
#   logs/RUN/<stage>/<law>/  that law's own run.log and card
#
# Monitor it with the viewer, following the one stream, for the whole cascade:
#
#   ./target/release/hff-watch --follow logs/RUN/stream.jsonl
#
# Every law appends to that file, so the viewer rolls onto the next law on its
# own. It is never restarted and never repointed.
set -u
run_all() {
ROOT=/Users/andrewmorgan/Dev/gamakon/fuller
DS=/Users/andrewmorgan/Dev/gamakon/hff/notebooks/_ledgers/pmlb_repo/datasets
SP=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/b3c1f1fe-1035-445f-ada0-9f05e732d5e6/scratchpad
RUN=$ROOT/logs/RUN
mkdir -p $SP/data $RUN/solved
# The one stream. Truncated ONCE here, appended by every fit after.
: > $RUN/stream.jsonl
echo "CASCADE START $(date)" > $RUN/sweep.log

for st in 60 120 180; do
  echo "=== STAGE ${st}s ===" >> $RUN/sweep.log
  while read n; do
    [ -f $RUN/solved/$n ] && continue
    [ -f $SP/data/$n.tsv ] || zcat < $DS/$n/$n.tsv.gz > $SP/data/$n.tsv 2>/dev/null || continue
    O=$RUN/${st}s/$n; mkdir -p $O
    HFF_TELEMETRY_APPEND=1 \
    EVOLVE_TYPED_DEPTH=2 EVOLVE_POP_INTAKE=2000 EVOLVE_POP_CHAMPION=2000 \
    EVOLVE_MAX_GENERATIONS=500000 EVOLVE_SECONDS=$st \
    EVOLVE_PUMP_EVERY=100 EVOLVE_COHORT_MERGE=10000 EVOLVE_GENE_SUBSETS=1 \
    EVOLVE_SEED=7014 EVOLVE_PROGRESS_EVERY=5 \
    EVOLVE_TELEMETRY_FILE=$RUN/stream.jsonl EVOLVE_TELEMETRY_RUN_ID=${st}s_$n \
    nice -n 5 $ROOT/target/release/examples/evolve_fit $SP/data/$n.tsv > $O/run.log 2>&1
    g=$(grep -E '^GENERATIONS' $O/run.log | awk '{print $2}')
    r=$(grep 'R² on the unseen test rows:' $O/run.log | awk '{print $NF}')
    if grep -q "early_stop" $O/run.log 2>/dev/null; then
      touch $RUN/solved/$n
      printf 'STOP  %-26s %5ss gen %-7s testR2 %s\n' "$n" "$st" "$g" "${r:--}" >> $RUN/sweep.log
    else
      printf '      %-26s %5ss gen %-7s testR2 %s\n' "$n" "$st" "$g" "${r:--}" >> $RUN/sweep.log
    fi
  done < $ROOT/experiments/order133.txt
done
echo "CASCADE DONE $(date)" >> $RUN/sweep.log
}
run_all
