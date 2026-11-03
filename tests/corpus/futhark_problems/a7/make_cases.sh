#!/bin/bash
# The stored cases of A7 (the matrix kingdom as a problem of futhark_evolve): the interface's nine matrices, one a
# class in the view's class order, from `futhark dataset` (seeded, values uniform in [-8, 8]), and the yardstick's
# output on them through `futhark c` (the published kernel: the product computed twice). Every role is at least 1.
#   bash tests/corpus/futhark_problems/a7/make_cases.sh
set -eu
cd "$(dirname "$0")"
B=$(mktemp -d)
futhark c yardstick.fut -o "$B/ys"
printf '# name\tsplit\tn\tm\tp\tseed (futhark dataset -s seed, f32 in [-8, 8]; the nine matrices m_n_n .. m_p_p in the class order; outputs by yardstick.fut through futhark c)\n' > cases.tsv
while read -r name split n m p seed; do
  i=0
  : > "$name.in"
  for shape in "$n $n" "$n $m" "$n $p" "$m $n" "$m $m" "$m $p" "$p $n" "$p $m" "$p $p"; do
    set -- $shape
    futhark dataset -b -s "$((seed * 16 + i))" --f32-bounds=-8:8 -g "[$1][$2]f32" >> "$name.in"
    i=$((i + 1))
  done
  "$B/ys" -b < "$name.in" > "$name.out"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$split" "$n" "$m" "$p" "$seed" >> cases.tsv
  echo "$name: n $n, m $m, p $p"
done <<'CASES'
train_n8_m8_p8_s1 train 8 8 8 1
train_n16_m12_p8_s2 train 16 12 8 2
train_n32_m32_p32_s3 train 32 32 32 3
held_n7_m5_p3_s4 held 7 5 3 4
held_n1_m1_p1_s5 held 1 1 1 5
test_n64_m64_p64_s6 test 64 64 64 6
CASES
rm -r "$B"
