#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
guard() { "$@" & local pid=$!; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null && echo "KILLED by 120 s guard"; wait $pid; }
guard $X $W/vblock.wgsl $W/tables.f32 $W/params_A.f32 1 $W/gpu_A_s1.f32
guard $X $W/vblock.wgsl $W/tables.f32 $W/params_A.f32 1 $W/gpu_A_s1_again.f32
guard $X $W/vblock.wgsl $W/tables.f32 $W/params_layer.f32 1 $W/gpu_layer.f32 50
guard $X $W/read.wgsl $W/nibbles_layer.u32 $W/params_layer.f32 1 $W/read_layer.f32 50
echo DONE
