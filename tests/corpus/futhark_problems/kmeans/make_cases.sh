#!/bin/bash
# The stored cases of the kmeans step: random points and centres from futhark-benchmarks' own generator
# (`futhark dataset`, seeded), both uniform in [0, 1], and one step of the published program on them through
# `futhark c` (the yardstick: the published assignment and update unchanged). Random centres leave some clusters
# empty, which the published update makes the zero vector.
#   bash tests/corpus/futhark_problems/kmeans/make_cases.sh
set -eu
cd "$(dirname "$0")"
B=$(mktemp -d)
futhark c yardstick.fut -o "$B/ys"
printf '# name\tsplit\tn\td\tk\tseed (futhark dataset -s seed; points and centres U[0,1]; outputs by yardstick.fut through futhark c)\n' > cases.tsv
while read -r name split n d k seed; do
  {
    futhark dataset -b -s "$((seed * 2))" --f32-bounds=0:1 -g "[$n][$d]f32"
    futhark dataset -b -s "$((seed * 2 + 1))" --f32-bounds=0:1 -g "[$k][$d]f32"
  } > "$name.in"
  "$B/ys" -b < "$name.in" > "$name.out"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$split" "$n" "$d" "$k" "$seed" >> cases.tsv
  echo "$name: n $n, d $d, k $k"
done <<'CASES'
train_n200_d4_k5_s1 train 200 4 5 1
train_n200_d4_k5_s2 train 200 4 5 2
train_n500_d8_k10_s3 train 500 8 10 3
held_n777_d6_k7_s4 held 777 6 7 4
held_n1_d1_k1_s5 held 1 1 1 5
CASES
rm -r "$B"
