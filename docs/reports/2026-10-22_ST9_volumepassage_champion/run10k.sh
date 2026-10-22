#!/bin/zsh
B=/private/tmp/claude-501/st9p2; L=/private/tmp/claude-501/locks
until mkdir $L/gpu 2>/dev/null; do sleep 3; done
cd /Users/andrewmorgan/Dev/gamakon/phylu
echo "## $(date -u +%T) start; load $(uptime | sed 's/.*averages: //'); fuller $(git -C ../fuller rev-parse --short HEAD) phylu $(git rev-parse --short HEAD)"
$B/guard.sh 108000 $B/wgsl_true_image_fit_run10k --shader volumepassage --size 256 --pop 32 --generations 10000 --seed 1 --pump-every 50 --checkpoint-every 500 --chart-every 500 --out $B/run10k
echo "## $(date -u +%T) exit $?"
rmdir $L/gpu
