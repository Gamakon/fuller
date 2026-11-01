#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
export VBLOCK_INNER=20
for k in row_gen_v4 row_gen_v5 row_read_v4; do
  T=tables_v3.f32; [ $k = row_read_v4 ] && T=nibbles_layer.u32
  echo "== $k"; $X $W/$k.wgsl $W/$T $W/params_layer13.f32 1013 $W/c_$k.f32 50 1 10624 | grep median
done
