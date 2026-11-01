#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
for k in full v3; do
  echo "== $k"; $X $W/abl_$k.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/abl_$k.f32 100 8 | grep median
done
echo "== read"; $X $W/fused_read_bpw.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/fr.f32 100 8 | grep median
