#!/bin/bash
cd /Users/andrewmorgan/Dev/gamakon/fuller
source tmp/fixf/gpu_lock.sh
take_gpu
W=experiments/virtual_blocks/wgsl; X=target/release/examples/vblock_wgsl
echo "== v4 unfused (quality, half A)"; $X $W/vblock_v4.wgsl $W/tables_v3.f32 $W/params_A.f32 1 $W/q_vblock_v4.f32
echo "== v4 unfused layer"; $X $W/vblock_v4.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/v4_layer13_s1013.f32 300 | grep median
echo "== v3 unfused layer"; $X $W/vblock_v3.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/t3.f32 300 | grep median
echo "== row gen v4";  $X $W/row_gen_v4.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/rg4.f32 300 1 10624 | grep median
echo "== row read v4"; $X $W/row_read_v4.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/rr4.f32 300 1 10624 | grep median
echo "== row gen v3";  $X $W/row_gen.wgsl $W/tables_v3.f32 $W/params_layer13.f32 1013 $W/rg.f32 300 1 10624 | grep median
echo "== row read";    $X $W/row_read.wgsl $W/nibbles_layer.u32 $W/params_layer13.f32 1 $W/rr.f32 300 1 10624 | grep median
