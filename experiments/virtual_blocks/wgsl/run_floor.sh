#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
echo "== floor: read kernel, 1 row";  $X $W/row_read_v4.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/fl.f32 300 1 1 | grep median
echo "== floor: gen kernel, 1 row";   $X $W/row_gen_v4.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/fl2.f32 300 1 1 | grep median
