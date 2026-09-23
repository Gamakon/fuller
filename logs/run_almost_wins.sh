#!/bin/zsh
# The 7 almost-wins, 3 seeds each, new rules live. Starts when parity is done.
H=/Users/andrewmorgan/Dev/gamakon/hff/notebooks
while pgrep -f run_all_tests.sh >/dev/null; do sleep 10; done
cd $H
echo "######## 13-problem sweep, HFF_GPU=1 $(date '+%H:%M:%S')"
S=$(date +%s)
HFF_GPU=1 python3 _sweep_equation_recovery.py --problems I_11_19,I_12_1,I_12_2,I_12_4,I_12_5,I_13_4,I_14_3,I_14_4,I_15_3x,I_18_4,I_25_13,I_29_4,I_8_14 --parallel 4 --audit-mode sr_logs/t13rules_gpu > sr_logs/t13rules_gpu.log 2>&1
echo "wall=$(( $(date +%s)-S ))s"
python3 sr_report.py --dir sr_logs/t13rules_gpu 2>&1 | tail -20
echo "warning/error lines: $(grep -c 'Warning\|Traceback\|Error' sr_logs/t13rules_gpu.log)"
for seed in 5 11 23; do
  echo "######## seed $seed $(date '+%H:%M:%S')"
  S=$(date +%s)
  HFF_GPU=1 HFF_SEED=$seed python3 _sweep_equation_recovery.py --problems III_4_33,II_11_28,II_38_14,I_13_12,I_15_3t,I_48_2,test_12 --parallel 4 --audit-mode sr_logs/almost_s$seed > sr_logs/almost_s$seed.log 2>&1
  echo "wall=$(( $(date +%s)-S ))s"
  python3 sr_report.py --dir sr_logs/almost_s$seed 2>&1 | tail -16
  echo "warning/error lines: $(grep -c 'Warning\|Traceback\|Error' sr_logs/almost_s$seed.log)"
done
echo "######## ALL DONE"
