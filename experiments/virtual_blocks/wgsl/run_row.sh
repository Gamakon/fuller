#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
echo "== row gen";  $X $W/row_gen.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/rg.f32 300 1 10624 | grep median
echo "== row read"; $X $W/row_read.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/rr.f32 300 1 10624 | grep median
