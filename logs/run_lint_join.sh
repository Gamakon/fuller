#!/bin/zsh
# The linter in the join: 13-problem set (must stay 13/13), then one long
# evolved-only problem under lint / egglog / off for the join's own timing line.
H=/Users/andrewmorgan/Dev/gamakon/hff/notebooks
cd $H
P13=I_11_19,I_12_1,I_12_2,I_12_4,I_12_5,I_13_4,I_14_3,I_14_4,I_15_3x,I_18_4,I_25_13,I_29_4,I_8_14
for mode in lint egglog; do
  echo "######## 13-problem set, HFF_FULLER=$mode  $(date '+%H:%M:%S')"
  S=$(date +%s)
  HFF_FULLER=$mode HFF_GPU=1 python3 _sweep_equation_recovery.py --problems $P13 --parallel 4 --audit-mode sr_logs/t13_$mode > sr_logs/t13_$mode.log 2>&1
  echo "wall=$(( $(date +%s)-S ))s"
  python3 sr_report.py --dir sr_logs/t13_$mode 2>&1 | sed -n 2,4p
  echo "warning/error lines: $(grep -c 'Warning\|Traceback\|Error' sr_logs/t13_$mode.log)"
done
for mode in lint egglog off; do
  echo "######## II_38_14 evolved-only, HFF_FULLER=$mode  $(date '+%H:%M:%S')"
  S=$(date +%s)
  HFF_FULLER=$mode HFF_GPU=1 HFF_RULES=none python3 _sweep_equation_recovery.py --problems II_38_14 --parallel 1 --audit-mode sr_logs/one_$mode > sr_logs/one_$mode.log 2>&1
  echo "wall=$(( $(date +%s)-S ))s"
  grep -h "^\[join\]" sr_logs/one_$mode/II_38_14.run.log | cut -c1-330
  echo "warning/error lines: $(grep -c 'Warning\|Traceback\|Error' sr_logs/one_$mode.log)"
done
echo "######## ALL DONE"
