#!/bin/zsh
# Parity on the simplify + trigsimp corpora, run to completion. 3-hour guard.
cd /Users/andrewmorgan/Dev/gamakon/fuller
for c in simplify trigsimp; do
  echo "=== $c start $(date)"
  S=$(date +%s)
  ./target/release/parity parity/corpus/$c.jsonl & PID=$!
  for i in $(seq 1 10800); do kill -0 $PID 2>/dev/null || break; sleep 1; done
  if kill -0 $PID 2>/dev/null; then kill -9 $PID; echo "=== $c KILLED at 3h"; fi
  echo "=== $c done in $(( $(date +%s) - S )) s"
done
