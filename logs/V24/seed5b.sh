#!/bin/zsh
# V24 SEEDS FOR v24_ingest5b (2026-10-17): 32 fresh model-written designs for
# the microbatch-keys-only contract (problems/v24_ingest5b/contract.md, its
# section "v24_ingest5b: what binding 0 holds"), the method of run 1's seeds
# (phylu wgsl_fit --seed-from model --seed-models sonnet:16,opus:8,codex:4,fable:4):
# fuller examples/v24_seed.rs, the same contract text, strategies and
# variation keys, the calls in parallel, a failed or non-compiling seed asked
# again of its model (three calls at most). Model calls only: no GPU, no CPU
# executor; each reply checked by naga on the host (harness::kernel_info).
# Out: logs/V24/seeds5b/ (seedNN.wgsl, models.tsv for --seed-from saved
# --seed-dir, calls/, checks.tsv); log logs/V24/seed5b.log.
# Guard: the whole job is killed after 3,600 s (run 1's slowest call, Fable,
# took 760 s; three rounds at most).
F=/Users/andrewmorgan/Dev/gamakon/fuller; OUT=$F/logs/V24/seeds5b; LOG=$F/logs/V24/seed5b.log; B=/tmp/claude-501/b/seed5b
set -e
if pgrep -x cargo >/dev/null || pgrep -x rustc >/dev/null; then echo "cargo or rustc is running: wait for the build window"; exit 1; fi
echo "[$(date -u +%H:%M:%SZ)] build v24_seed at fuller $(git -C $F rev-parse --short HEAD)" | tee -a $LOG
(cd $F && cargo build --release --features gpu,wgsl --example v24_seed) >> $LOG 2>&1
mkdir -p $B && cp $F/target/release/examples/v24_seed $B/v24_seed
echo "[$(date -u +%H:%M:%SZ)] start: 32 seeds into $OUT" | tee -a $LOG
nohup $B/v24_seed --problem $F/problems/v24_ingest5b --out $OUT --models sonnet:16,opus:8,codex:4,fable:4 --seed 5 >> $LOG 2>&1 &
PID=$!
( for i in $(seq 1 3600); do kill -0 $PID 2>/dev/null || exit 0; sleep 1; done; echo "[$(date -u +%H:%M:%SZ)] guard: killed after 3600 s" >> $LOG; kill -9 $PID 2>/dev/null ) &
echo "pid $PID; tail -f $LOG"
