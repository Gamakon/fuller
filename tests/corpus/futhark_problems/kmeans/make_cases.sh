#!/bin/bash
# THE MAGNITUDE CASES of the k-means problem (docs/PLAN_futhark_kingdom.md 4c item 3): random points and starting
# centres from futhark-benchmarks' own generator (`futhark dataset`, seeded), NEVER quantised, at the scales 10^-3, 1,
# 10^3 and 10^5 (the `--f32-bounds` range), one case with mixed scales within it (coordinate t scaled by
# 10^((t mod 7) - 3), shape_cases.fut `mixed`), one with very unequal cluster sizes (points in [0, 1], centres in
# [0, 2]: the centres outside the cube take few or no points), and one built to break an uncompensated sum (the FIRST
# 4 of 2,000 points scaled by 10^4 among points around 1, two centres in [0, 1]: a cluster sums the big ones first and
# then a thousand small ones, each rounding at the big sum's ulp; shape_cases.fut `breaking`). n, d and k are small enough for the host interpreter. INPUTS ONLY: k-means is scored on
# its clustering objective, run to UltraDim's stopping rule (docs/KMEANS_FITNESS.md), so there is no answer to store.
# reference64.fut (UltraDim's step in f64) is a check of the seed's translation, not a stored output: a Rust test
# compiles and runs it on every case here against the Rust f64 step.
#   bash tests/corpus/futhark_problems/kmeans/make_cases.sh
set -eu
cd "$(dirname "$0")"
B=$(mktemp -d)
futhark c shape_cases.fut -o "$B/shape"
printf '# name\tsplit\tn\td\tk\tseed\tscale\tshape (futhark dataset -s seed; points and centres U[0, scale], never quantised; shape: plain, mixed = coordinate t scaled by 10^((t mod 7) - 3), unequal = centres U[0, 2 scale], breaking = the first 4 points scaled by 10^4; inputs only, no output is stored: k-means is scored on its clustering objective run to UltraDim stopping rule, docs/KMEANS_FITNESS.md)\n' > cases.tsv
while read -r name split n d k seed scale shape; do
  cscale=$scale
  if [ "$shape" = unequal ]; then cscale=$(python3 -c "print(2 * $scale)"); fi
  {
    futhark dataset -b -s "$((seed * 2))" --f32-bounds=0:"$scale" -g "[$n][$d]f32"
    futhark dataset -b -s "$((seed * 2 + 1))" --f32-bounds=0:"$cscale" -g "[$k][$d]f32"
  } > "$B/draw"
  case "$shape" in
    plain|unequal) cp "$B/draw" "$name.in" ;;
    mixed) "$B/shape" -e mixed -b < "$B/draw" > "$name.in" ;;
    breaking) { printf 'b\x02\x00 i64'; printf '\x04\x00\x00\x00\x00\x00\x00\x00'; cat "$B/draw"; } | "$B/shape" -e breaking -b > "$name.in" ;;
    *) echo "unknown shape $shape" >&2; exit 1 ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$split" "$n" "$d" "$k" "$seed" "$scale" "$shape" >> cases.tsv
  echo "$name: n $n, d $d, k $k, scale $scale, $shape"
done <<'CASES'
train_s1_n200_d4_k5 train 200 4 5 1 1 plain
train_s1e-3_n200_d4_k5 train 200 4 5 2 0.001 plain
train_s1e3_n300_d6_k8 train 300 6 8 3 1000 plain
train_s1e5_n200_d4_k5 train 200 4 5 4 100000 plain
train_breaking_n2000_d4_k2 train 2000 4 2 7 1 breaking
held_mixed_n400_d7_k6 held 400 7 6 5 1 mixed
held_unequal_n500_d3_k6 held 500 3 6 11 1 unequal
test_s1_n300_d5_k4 test 300 5 4 8 1 plain
CASES
rm -r "$B"
