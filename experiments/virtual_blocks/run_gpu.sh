#!/bin/bash
# The GPU baseline run, under the machine's GPU lock, with a kill guard (900 s: compile + 3 timed sets).
#   bash experiments/virtual_blocks/run_gpu.sh WORK SET[:t] ...
set -u
source /Users/andrewmorgan/Dev/gamakon/fuller/tmp/fixf/gpu_lock.sh
take_gpu
# The bench moved to phylu with the Futhark evaluator (fuller docs/PLAN_futhark_kingdom.md 5b):
#   cd ../phylu && cargo build --release --example virtual_blocks_bench
bin=/Users/andrewmorgan/Dev/gamakon/phylu/target/release/examples/virtual_blocks_bench
( "$bin" "$@" & P=$!; for i in $(seq 1 900); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null; wait $P ) &
wait $!
