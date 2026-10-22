#!/bin/zsh
# UI2 GPU jobs, one at a time, each under its own guard; the lock taken by mkdir, released only by us.
LOCK=/private/tmp/claude-501/locks/gpu
F=/Users/andrewmorgan/Dev/gamakon/fuller; S=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/a64dc98c-8739-45f5-92d5-fe1f37ac1d88/scratchpad; OUT=$F/logs/V24/ui2
st() { date -u +%H:%M:%SZ; }
echo "[$(st)] waiting for the GPU lock"
until mkdir $LOCK 2>/dev/null; do sleep 10; done
echo "ui2-gpu pid $$" > $LOCK/owner
trap 'rm -f $LOCK/owner; rmdir $LOCK; echo "[$(st)] GPU lock released"' EXIT
echo "[$(st)] GPU lock taken"
job() { # name, guard seconds, command...
  local name=$1 g=$2; shift 2
  echo "[$(st)] start $name (guard $g s)"
  "$@" > $OUT/$name.log 2>&1 &
  local pid=$!
  ( sleep $g; kill -9 $pid 2>/dev/null && echo "[$(st)] GUARD killed $name" ) &
  local gp=$!
  wait $pid; local rc=$?
  kill $gp 2>/dev/null
  echo "[$(st)] end $name rc $rc"
  return $rc
}
cd $F
job smoke 600 $S/v24_ingest250k_test --exact v24_ingest250k_reference_runs_stage_1_on_the_device --nocapture || exit 1
job scan_budget_s1 600 $S/v24_measure --problem problems/v24_ingest250k --kernel logs/V24/run4/best.wgsl --runs 3 --set stages=1 || exit 1
job scan_wide_s1 600 $S/v24_measure --problem problems/v24_ingest250k --kernel logs/V24/ui2/run4_champion_rows262144.wgsl --runs 3 --set stages=1 --set arena_u32=33554432 || exit 1
job scan_budget_s5 1800 $S/v24_measure --problem problems/v24_ingest250k --kernel logs/V24/run4/best.wgsl --runs 3 || exit 1
job scan_wide_s5 1800 $S/v24_measure --problem problems/v24_ingest250k --kernel logs/V24/ui2/run4_champion_rows262144.wgsl --runs 3 --set arena_u32=33554432 || exit 1
echo "[$(st)] all done"
