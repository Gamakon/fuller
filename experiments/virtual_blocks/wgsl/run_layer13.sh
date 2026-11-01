#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl
target/release/examples/vblock_wgsl $W/${KERNEL:-vblock}.wgsl $W/${TABLES:-tables}.f32 $W/params_layer${LAYER:-13}.f32 ${SEED:-1} $W/${PREFIX:-gpu}_layer${LAYER:-13}_s${SEED:-1}.f32 &
P=$!; for i in $(seq 1 120); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null; wait $P
