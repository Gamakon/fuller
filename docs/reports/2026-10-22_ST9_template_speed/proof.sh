#!/bin/zsh
# The hang reproducer (docs/reports/2026-10-21_metal_hang_k2) with the shadow-counter loop spelling, alone under the GPU lock,
# every process time-limited. Order: shadow (Apple compiler, Metal API only), shadow through wgpu, the loop_init control
# (expected to hang, killed at 8 s), then the shadow through wgpu again as the probe that the GPU is free.
L=/private/tmp/claude-501/locks/gpu; H=/private/tmp/claude-501/st9p8/hang; R=/Users/andrewmorgan/Dev/gamakon/fuller/docs/reports/2026-10-21_metal_hang_k2
LOOP=/private/tmp/claude-501/st9p8/target/release/examples/wgsl_true_loop
until mkdir $L 2>/dev/null; do sleep 3; done
g() { local s=$1; shift; "$@" & local p=$!; for i in $(seq 1 $s); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -9 $p 2>/dev/null; wait $p; echo "## exit $? $(date -u +%H:%M:%SZ)"; }
echo "## $(date -u +%H:%M:%SZ) start; load $(uptime | sed 's/.*averages: //'); WindowServer $(pgrep -x WindowServer)"
echo "## shadow, Metal API only"; g 20 $H/run_msl $H/shadow.metal 8
echo "## shadow, wgpu (naga + pipeline + one run, Gram bindings)"; g 30 $LOOP --file $H/gene583_reduced_shadow_loop.wgsl --stage all
echo "## control: the loop_init form, Metal API only (expected HANG)"; g 20 $H/run_msl $R/naga_msl_reduced.metal 8
echo "## control: naga's MSL of the for-form reduced kernel, wgpu options (expected HANG)"; g 20 $H/run_msl $H/control_for.metal 8
echo "## probe: shadow, wgpu again"; g 60 $LOOP --file $H/gene583_reduced_shadow_loop.wgsl --stage all
echo "## $(date -u +%H:%M:%SZ) end; load $(uptime | sed 's/.*averages: //'); WindowServer $(pgrep -x WindowServer)"
rmdir $L
