#!/bin/bash
# The stored cases of the kmeans step: random points and centres from futhark-benchmarks' own generator
# (`futhark dataset`, seeded), both uniform in [0, 1] and then QUANTISED to multiples of 2^-6 in [0, 1) by
# `quantise.fut` (floor(u * 64) / 64, 64 levels; see that file), and one step of the published program on them through
# `futhark c` (the yardstick: the published assignment and update unchanged). Random centres leave some clusters
# empty, which the published update makes the zero vector.
#
# Why dyadic (decided 2026-11-03, `docs/PLAN_futhark_kingdom.md` 4a): the comparison policy is bit for bit with no
# tolerance anywhere, and the GPU adds a cluster's points and a distance's squares in another order than `futhark c`
# and the host interpreter; on the uniforms of 24-bit mantissas 65-81 % of the cells differed by one or two f32 bits
# on 4 of the 5 cases. With every input a multiple of 2^-6 every partial sum of every order is exact in f32: a
# cluster sum is at most n < 2^18 terms of granularity 2^-6 (n * 2^6 < 2^24 granules), a squared distance at most
# d <= 2^12 terms of granularity 2^-12 (d * 2^12 < 2^24 granules); the only roundings left are the final divisions by
# the counts, single correctly-rounded operations on every IEEE path. `tests_kmeans::every_stored_case_is_dyadic...`
# proves the property on the stored bytes.
#   bash tests/corpus/futhark_problems/kmeans/make_cases.sh
set -eu
cd "$(dirname "$0")"
B=$(mktemp -d)
futhark c yardstick.fut -o "$B/ys"
futhark c quantise.fut -o "$B/q"
printf '# name\tsplit\tn\td\tk\tseed (futhark dataset -s seed, points and centres U[0,1], quantised by quantise.fut to multiples of 2^-6 in [0,1) so every sum of every order is exact in f32; outputs by yardstick.fut through futhark c)\n' > cases.tsv
while read -r name split n d k seed; do
  {
    futhark dataset -b -s "$((seed * 2))" --f32-bounds=0:1 -g "[$n][$d]f32"
    futhark dataset -b -s "$((seed * 2 + 1))" --f32-bounds=0:1 -g "[$k][$d]f32"
  } | "$B/q" -b > "$name.in"
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
