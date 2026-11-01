#!/bin/bash
# The GPU lock is taken ONCE for the whole batch; arguments go to candidate_runner.py.
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
run_bg python3 experiments/virtual_blocks/wgsl/candidate_runner.py "$@"
