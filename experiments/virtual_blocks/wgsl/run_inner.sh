#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
export VBLOCK_INNER=20
echo "== row read v4"; $X $W/row_read_v4.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/rr4.f32 50 1 10624 | grep median
echo "== row gen v4";  $X $W/row_gen_v4.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/rg4.f32 50 1 10624 | grep median
echo "== row read v3 layout"; $X $W/row_read.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/rr.f32 50 1 10624 | grep median
echo "== row read fast"; $X $W/row_read_fast.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/rrf.f32 50 1 10624 | grep median
echo "== floor 1 row"; $X $W/row_read_v4.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/fl.f32 50 1 1 | grep median
