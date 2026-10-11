#!/bin/zsh
# One engine run of the safeWGSL kingdom. Usage: [DUMPS=safe96/karva] run_engine.sh <name> <gens> <seed> <seconds> [extra args]
# Stop it with: rm logs/HGP/001/<name>/killme.trigger   (never a signal)
set -u
name=$1; gens=$2; seed=$3; secs=$4; shift 4
cd /Users/andrewmorgan/Dev/gamakon/fuller
out=logs/HGP/001/$name
mkdir -p "$out"
if pgrep -f "examples/wgsl_fit" >/dev/null; then echo "another fit is running; refusing"; exit 1; fi
echo "[$(date -u +%H:%M:%SZ)] start $name gens=$gens seed=$seed seconds=$secs $*"
../phylu/target/release/examples/wgsl_fit --dumps "$PWD/logs/HGP/001/${DUMPS:-safe/karva}" \
  --pop-intake 20 --pop-champion 10 --gens "$gens" --seed "$seed" --elites 2 --pump-every 5 --cohort-merge 20 \
  --point-mutations 2 --llm 4 --proposals 2 --model sonnet --seconds "$secs" --out "$PWD/$out" "$@" &
pid=$!
# Guard: the budget plus ten minutes; past it the run file is deleted, never a kill.
limit=$((secs + 600))
for i in $(seq 1 $limit); do kill -0 $pid 2>/dev/null || break; sleep 1; done
if kill -0 $pid 2>/dev/null; then echo "[$(date -u +%H:%M:%SZ)] guard: deleting the run file"; rm -f "$out/killme.trigger"; wait $pid; fi
echo "[$(date -u +%H:%M:%SZ)] end $name exit $?"
