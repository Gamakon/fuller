#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
guard() { "$@" & local pid=$!; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null && echo "KILLED by 120 s guard"; wait $pid; }
for k in vblock_sg_fix1.wgsl:tables_fast.f32 vblock_sg_fix2.wgsl:tables_fast.f32 vblock_sg_fix4.wgsl:tables_fast.f32; do
  K=${k%%:*}; T=${k##*:}; echo "== $K"
  guard $X $W/$K $W/$T $W/params_A.f32 1 $W/q_${K%.wgsl}.f32
  guard $X $W/$K $W/$T $W/params_layer13.f32 1013 $W/t_${K%.wgsl}.f32 100 | grep median
done
echo "== read"; guard $X $W/read.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/read_layer.f32 100 | grep median
