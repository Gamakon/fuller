#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
guard() { "$@" & local pid=$!; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null && echo "KILLED by 120 s guard"; wait $pid; }
for bpw in 1 8 32 128; do
  echo "== bpw $bpw gen";  guard $X $W/fused_gen_bpw.wgsl $W/tables_fast.f32 $W/params_layer13.f32 1013 $W/fg_$bpw.f32 100 $bpw | grep median
  echo "== bpw $bpw read"; guard $X $W/fused_read_bpw.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/fr_$bpw.f32 100 $bpw | grep median
done
