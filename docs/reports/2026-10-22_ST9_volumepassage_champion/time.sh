#!/bin/zsh
B=/private/tmp/claude-501/st9time; L=/private/tmp/claude-501/locks
until mkdir $L/gpu 2>/dev/null; do sleep 3; done
echo "## $(date -u +%T) start; load $(uptime | sed 's/.*averages: //'); fuller $(git -C /Users/andrewmorgan/Dev/gamakon/fuller rev-parse --short HEAD)"
cd /Users/andrewmorgan/Dev/gamakon/fuller
/private/tmp/claude-501/st9p2/guard.sh 1200 $B/st9_champion_time --hof /private/tmp/claude-501/st9p2/run10k/hof.bin --rounds 61 --sizes 256,512 --seed 1 --out $B/out
echo "## $(date -u +%T) exit $?; load $(uptime | sed 's/.*averages: //'); WindowServer $(pgrep -x WindowServer)"
rmdir $L/gpu
