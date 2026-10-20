#!/bin/zsh
# V24 run 6 (2026-10-20): the WGSL kingdom on the rev-4 typed core, seeded from logs/V24/seeds5b on
# problems/v24_ingest5b, one island per declared interface, gene operators only, no model calls.
# Two arms, one after the other: U = uniform heads (N-1)*a_T (design guarantee), S = heads from the largest seed per type.
G=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/a64dc98c-8739-45f5-92d5-fe1f37ac1d88/scratchpad/g4
B=$G/wgsl_typed_bin/wgsl_fit; F=/Users/andrewmorgan/Dev/gamakon/fuller
echo "[$(date -u +%H:%M:%SZ)] binary $(shasum -a 256 $B | cut -c1-16) built at $(cat $G/wgsl_typed_bin.COMMIT)"
cd /Users/andrewmorgan/Dev/gamakon/phylu
for arm in uniform largest; do
  OUT=$F/logs/V24/run6_$arm; [[ -e $OUT ]] && { echo "$OUT exists"; exit 2 }
  echo "[$(date -u +%H:%M:%SZ)] start arm $arm"
  ( $B --problem $F/problems/v24_ingest5b --typed true --typed-heads $arm --seed-from saved --seed-dir $F/logs/V24/seeds5b \
      --layer7 arena+rows --pop-intake 60 --pop-champion 60 --gens 20 --elites 2 --wild-point-rate 0.02 \
      --llm 0 --repairs 0 --model-cross 0 --proposals 0 --out $OUT > $F/logs/V24/run6_$arm.log 2>&1 & P=$!
    for i in $(seq 1 3600); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null; wait $P; echo "[$(date -u +%H:%M:%SZ)] end arm $arm: exit $?" )
done
