#!/bin/zsh
# guard.sh SECONDS cmd... : run cmd, kill it after SECONDS
secs=$1; shift
"$@" &
pid=$!
for i in $(seq 1 $secs); do kill -0 $pid 2>/dev/null || break; sleep 1; done
if kill -0 $pid 2>/dev/null; then echo "GUARD: killed after $secs s"; kill -9 $pid; fi
wait $pid 2>/dev/null
