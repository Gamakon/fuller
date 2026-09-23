#!/bin/zsh
# Frozen live-population corpus: every expression the join asks fuller to
# expand, across 7 long evolved-only runs. Then egglog's smallest_form on it.
F=/Users/andrewmorgan/Dev/gamakon/fuller
H=/Users/andrewmorgan/Dev/gamakon/hff/notebooks
RAW=$F/logs/live_raw.tsv
: > $RAW
cd $H
S=$(date +%s)
FULLER_DUMP_MATH=$RAW HFF_GPU=1 HFF_RULES=none python3 _sweep_equation_recovery.py --problems III_4_33,II_11_28,II_38_14,I_13_12,I_15_3t,I_48_2,test_12 --parallel 4 --audit-mode sr_logs/live_capture > sr_logs/live_capture.log 2>&1
echo "capture wall=$(( $(date +%s)-S ))s  lines=$(wc -l < $RAW)"
python3 sr_report.py --dir sr_logs/live_capture 2>&1 | sed -n 2,3p
cd $F
python3 - <<'PY'
import collections
raw="/Users/andrewmorgan/Dev/gamakon/fuller/logs/live_raw.tsv"
c=collections.Counter(l.rstrip("\n") for l in open(raw) if l.strip())
items=sorted(c.items())
open("logs/live_exprs.tsv","w").write("".join("\t".join(k.split("\t")[:2])+"\n" for k,_ in items))
open("logs/live_psets.txt","w").write("".join(k.split("\t")[2]+"\n" for k,_ in items))
open("logs/live_counts.txt","w").write("".join(str(v)+"\n" for _,v in items))
print("distinct expressions",len(items),"of",sum(c.values()))
PY
S=$(date +%s)
./target/release/examples/measure_smallest_form logs/live_exprs.tsv > logs/live_smallest_form.tsv
echo "egglog smallest_form on the live corpus: $(( $(date +%s)-S ))s"
echo "######## ALL DONE"
