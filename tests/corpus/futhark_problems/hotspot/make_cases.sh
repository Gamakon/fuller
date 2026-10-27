#!/bin/bash
# The stored cases of hotspot: random inputs from futhark-benchmarks' own generator (`futhark dataset`, seeded) and
# the published program's outputs on them through `futhark c` (the yardstick, the published kernel unchanged).
# Temperatures uniform in [320, 340], power uniform in [0, 0.005] (the published data's range: ambient 80, chips
# running at a few hundred kelvin). Every dimension is at least 2 (the published kernel's precondition).
#   bash tests/corpus/futhark_problems/hotspot/make_cases.sh
set -eu
cd "$(dirname "$0")"
B=$(mktemp -d)
futhark c yardstick.fut -o "$B/ys"
printf '# name\tsplit\trows\tcols\titerations\tseed (futhark dataset -s seed; temperatures U[320,340], power U[0,0.005]; outputs by yardstick.fut through futhark c)\n' > cases.tsv
while read -r name split rows cols iters seed; do
  {
    futhark dataset -b -s "$seed" --i32-bounds="$iters:$iters" -g i32
    futhark dataset -b -s "$((seed * 2))" --f32-bounds=320:340 -g "[$rows][$cols]f32"
    futhark dataset -b -s "$((seed * 2 + 1))" --f32-bounds=0:0.005 -g "[$rows][$cols]f32"
  } > "$name.in"
  "$B/ys" -b < "$name.in" > "$name.out"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$split" "$rows" "$cols" "$iters" "$seed" >> cases.tsv
  echo "$name: ${rows}x${cols}, $iters iterations"
done <<'CASES'
train_64x64_i10_s1 train 64 64 10 1
train_64x64_i10_s2 train 64 64 10 2
train_32x100_i5_s3 train 32 100 5 3
held_96x128_i7_s4 held 96 128 7 4
held_2x2_i3_s5 held 2 2 3 5
CASES
rm -r "$B"
