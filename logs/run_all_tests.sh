#!/bin/zsh
# Everything together, to completion. Parity last (longest), 3 h guard per corpus.
F=/Users/andrewmorgan/Dev/gamakon/fuller
H=/Users/andrewmorgan/Dev/gamakon/hff/notebooks
step() { echo; echo "######## $1  $(date '+%H:%M:%S')"; }
cd $F
step "1 build python extension (python,gpu)"
S=$(date +%s); maturin develop --release --features python,gpu 2>&1 | grep -E "^error|^warning|Installed|Finished"; echo "secs=$(( $(date +%s)-S ))"
step "2 cargo test --features gpu"
S=$(date +%s); RUSTFLAGS="-D warnings" cargo test --release --features gpu 2>&1 | grep -E "test result|FAILED|panicked|^error"; echo "secs=$(( $(date +%s)-S ))"
step "3 fuller timings on 847 stored expressions"
cargo build --release --examples 2>&1 | grep -E "^error|^warning"
./target/release/examples/measure_smallest_form logs/hof_exprs.tsv > logs/smallest_form_all.tsv
python3 - <<'PY'
import json
g=[a for a in map(json.loads,open("/Users/andrewmorgan/Dev/gamakon/hff/notebooks/_ledgers/hof_dataset/genes.jsonl")) if a.get("math")]
r=[l.split("\t") for l in open("/Users/andrewmorgan/Dev/gamakon/fuller/logs/smallest_form_all.tsv")]
ok=[(a,x) for a,x in zip(g,r) if x[0]!="ERR"]
print("smallest_form: errors",len(r)-len(ok),"secs",round(sum(float(x[2]) for _,x in ok),2))
print("weighted nodes: in",sum(a["count"]*int(x[0]) for a,x in ok),"stored-baseline",sum(a["count"]*a["fuller_nodes"] for a,x in ok),"now",sum(a["count"]*int(x[1]) for a,x in ok))
print("smaller than stored baseline:",sum(int(x[1])<a["fuller_nodes"] for a,x in ok),"larger:",sum(int(x[1])>a["fuller_nodes"] for a,x in ok))
PY
for n in 64 8192; do ./target/release/examples/profile_candidates logs/hof_exprs.tsv $n | head -6; done
step "4 the 13-problem sweep (GPU join)"
cd $H
S=$(date +%s)
python3 _sweep_equation_recovery.py --problems I_11_19,I_12_1,I_12_2,I_12_4,I_12_5,I_13_4,I_14_3,I_14_4,I_15_3x,I_18_4,I_25_13,I_29_4,I_8_14 --parallel 4 --audit-mode sr_logs/t13rules > sr_logs/t13rules.log 2>&1
echo "wall=$(( $(date +%s)-S ))s" | tee sr_logs/t13rules.timing
python3 sr_report.py sr_logs/t13rules 2>&1 | tail -25
grep -c "Warning\|Traceback\|Error" sr_logs/t13rules.log
cd $F
step "5 parity, all corpora"
for c in powsimp radsimp ratsimp trigsimp simplify; do
  S=$(date +%s)
  ./target/release/parity parity/corpus/$c.jsonl & PID=$!
  for i in $(seq 1 10800); do kill -0 $PID 2>/dev/null || break; sleep 1; done
  if kill -0 $PID 2>/dev/null; then kill -9 $PID; echo "$c KILLED at 3h"; fi
  echo "$c secs=$(( $(date +%s)-S ))"
done
step "ALL DONE"
