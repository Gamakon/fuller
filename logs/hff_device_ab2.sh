#!/bin/zsh
set -u
DATA=/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/b3c1f1fe-1035-445f-ada0-9f05e732d5e6/scratchpad/bacres1.tsv
BIN=./target/release/examples/evolve_fit
for POP in 2000 20000; do
  for HOST in 0 1; do
    export EVOLVE_HFF_ON_HOST=$HOST EVOLVE_SEED=7015 EVOLVE_SECONDS=60 EVOLVE_POP_INTAKE=$POP
    where="device"; [ "$HOST" = "1" ] && where="host"
    echo "=== pop $POP, HFF on $where ==="
    $BIN "$DATA" 2>&1 | grep -E "^population |^seconds:"
  done
done
echo "AB2 DONE"
