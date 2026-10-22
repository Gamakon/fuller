#!/bin/zsh
T=/private/tmp/claude-501/st9p3/time; L=/private/tmp/claude-501/locks
until grep -q '^## .* exit' /private/tmp/claude-501/st9p3/run.log; do sleep 2; done
until mkdir $L/gpu 2>/dev/null; do sleep 1; done
echo "## $(date -u +%T) lock taken; load $(uptime | sed 's/.*averages: //')"
python3 -I $T/top5.py /private/tmp/claude-501/st9p3/run/hof.bin $T/hof_top5.bin
cp /Users/andrewmorgan/Dev/gamakon/fuller/target/release/examples/st9_champion_time $T/st9_champion_time
cd /Users/andrewmorgan/Dev/gamakon/fuller
( $T/st9_champion_time --hof $T/hof_top5.bin --rounds 61 --sizes 64,128 --seed 3 --out $T/out & P=$!; for i in $(seq 1 300); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null; wait $P; echo "## $(date -u +%T) timing exit $?" )
rmdir $L/gpu
echo "## $(date -u +%T) lock released"
