#!/bin/zsh
# UI2: the v24 test set at 250,000 records (+ 2,500 oracles), today's index on it, and the
# fuller harness case, through qdrant's own unchanged scripts (experiments/v24_ingest).
# The ultradim wheel links Metal, so the whole chain holds the GPU lock.
# Guard: 4 h for the chain (kill of the process group), lock released only if we took it.
PY=/Users/andrewmorgan/data/arm/venv/bin/python
Q=/Users/andrewmorgan/dev/minkymorgan/qdrant/experiments/v24_ingest
SET=/Users/andrewmorgan/data/arm/v24_test_250k
CASE=/Users/andrewmorgan/data/arm/v24_case_250k
LOCK=/private/tmp/claude-501/locks/gpu
st() { date -u +%H:%M:%SZ; }
echo "[$(st)] waiting for the GPU lock"
until mkdir $LOCK 2>/dev/null; do sleep 15; done
echo "ui2-make_250k pid $$" > $LOCK/owner
trap 'rm -f $LOCK/owner; rmdir $LOCK; echo "[$(st)] GPU lock released"' EXIT
echo "[$(st)] GPU lock taken"
( sleep 14400; echo "[$(st)] GUARD: 4 h passed, killing the chain"; pkill -9 -f "experiments/v24_ingest|prep_arm_test.py|index_test_families.py|count_today_index.py|export_harness_case.py"; kill -TERM $$ ) &
GUARD=$!
set -e
cd $Q
echo "[$(st)] phase 1: prep_arm_test (sample, encode, upsert, CPU exact truth)"
$PY prep_arm_test.py --out $SET --n-stream 250000 --n-oracle 2500
echo "[$(st)] phase 2: index_test_families (today's trellis build per family)"
$PY index_test_families.py --dir $SET
echo "[$(st)] phase 3: count_today_index (today's search per stage)"
$PY count_today_index.py --dir $SET
echo "[$(st)] phase 4: export_harness_case"
$PY export_harness_case.py --test-set $SET --out $CASE
kill $GUARD 2>/dev/null
echo "[$(st)] chain done"
