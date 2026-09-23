#!/bin/zsh
# Sixth and seventh arms: the linter offers EVERY option (all rule levels +
# data-justified prunes), each scored on data. Same problems, same seeds.
H=/Users/andrewmorgan/Dev/gamakon/hff/notebooks
cd $H
P=I_11_19,I_12_1,I_12_2,I_12_4,I_12_5,I_13_4,I_14_3,I_14_4,I_15_3x,I_18_4,I_25_13,I_29_4,I_8_14,I_18_12,I_18_14,I_34_1,I_39_11
for seed in 5 11 23; do
  for arm in "lintall improve 0" "lintall hff 1"; do
    set -- ${=arm}; graft=$2; pars=$3
    tag=ab2_lintall-${graft}-p${pars}_s$seed
    echo "######## $tag  $(date '+%H:%M:%S')"
    S=$(date +%s)
    HFF_SEED=$seed HFF_FULLER=lint HFF_LINT_EXACTNESS=finite HFF_GRAFT_MODE=$graft HFF_PARSIMONY=$pars HFF_GPU=1 HFF_RULES=none python3 _sweep_equation_recovery.py --problems $P --parallel 6 --audit-mode sr_logs/$tag > sr_logs/$tag.log 2>&1
    echo "wall=$(( $(date +%s)-S ))s  warn/err lines: $(grep -c 'Warning\|Traceback\|Error' sr_logs/$tag.log)"
  done
done
echo "######## ALL DONE"
