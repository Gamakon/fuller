#!/bin/zsh
# V24 run 5: the long evolution on problems/v24_ingest5b (all 25,000 records in five stages,
# keys projected on the device; ingest sees only its microbatch's keys, search none, so an
# exhaustive scan is impossible), from the 32 fresh seeds written to 5b's contract
# (logs/V24/seeds5b). Layer 7 arena+rows (run 3's best arm). Built at wgpu/naga 30.0.1 in the
# ONE measuring worktree per repo (scratchpad/g4), no new worktrees.
# Pure gene expression programming (Andrew 2026-10-19): after the elites every row is a new gene child
# (Karva crossover between same-interface parents, point mutation, random genes); no model calls.
# Usage: FU=<fuller sha> PH=<phylu sha> HF=<hff sha> [SECONDS_CAP=14400] [SLOTS=184] run4.sh
# Stop it by deleting $OUT/killme.trigger.
set -e
G=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/a64dc98c-8739-45f5-92d5-fe1f37ac1d88/scratchpad/g4
F=/Users/andrewmorgan/Dev/gamakon/fuller; OUT=$F/logs/V24/run5; T=$G/target
stamp() { date -u +%H:%M:%SZ }
: ${FU:?fuller sha} ${PH:?phylu sha} ${HF:?hff sha}; SECONDS_CAP=${SECONDS_CAP:-14400}; SLOTS=${SLOTS:-184}
[[ -e $OUT ]] && { echo "[$(stamp)] $OUT exists: move it aside first"; exit 2 }
git -C $G/fuller checkout -q --detach $FU; git -C $G/phylu checkout -q --detach $PH; git -C $G/hff checkout -q --detach $HF
echo "[$(stamp)] V24 run 5: fuller $(git -C $G/fuller log --oneline -1)"
echo "[$(stamp)]            phylu  $(git -C $G/phylu log --oneline -1)"
echo "[$(stamp)]            hff    $(git -C $G/hff log --oneline -1)"
while pgrep -x cargo >/dev/null || pgrep -x rustc >/dev/null; do sleep 5; done
echo "[$(stamp)] build"
(cd $G/phylu && CARGO_TARGET_DIR=$T RUSTFLAGS="-D warnings" cargo build --release --features gpu,wgsl --example wgsl_fit)
(cd $G/fuller && CARGO_TARGET_DIR=$T RUSTFLAGS="-D warnings" cargo build --release --features gpu,wgsl --example wgsl_safe_convert)
mkdir -p $G/run5_bin && cp $T/release/examples/wgsl_fit $T/release/examples/wgsl_safe_convert $G/run5_bin/
B=$G/run5_bin
echo "[$(stamp)] reference dump at $SLOTS slots"
(cd $G/fuller && $B/wgsl_safe_convert --problem $G/fuller/problems/v24_ingest5b --out $B/ref_safe --karva $B/ref_karva --head 32 --slots $SLOTS)
FLAGS=(--problem $G/fuller/problems/v24_ingest5b --dumps $B/ref_karva
  --seed-from saved --seed-dir $F/logs/V24/seeds5b --raw-seeds true --admit-differential false --layer7 arena+rows
  --variation mutation --karva-cross true --cross-kind section --pure-gep true --pump-random true --pump-every 1
  --typed-cross 0 --rewrites 0 --rule-all false --point-mutations 0
  --model-rewrite 0 --model-cross-beat 0 --swap-tries 0 --llm 0 --repairs 0 --model-cross 0 --proposals 0
  --library $OUT/library --out $OUT --seed 1 --pop-intake 24 --pop-champion 10 --gens 100000
  --seconds $SECONDS_CAP)
echo "[$(stamp)] common-sense check (host only)"
(cd $G/phylu && $B/wgsl_fit $FLAGS --out $G/run5_check --admission-check true) || { echo "[$(stamp)] the check failed: the run does not start"; exit 1 }
echo "[$(stamp)] start V24 run 5 ($SECONDS_CAP s cap): stop it by deleting $OUT/killme.trigger"
cd $G/phylu && caffeinate -i $B/wgsl_fit $FLAGS
echo "[$(stamp)] end V24 run 5: exit $?"
