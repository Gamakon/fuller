-- A map whose grid is over WebGPU's 65,535 blocks a dimension: 2^24 elements at 256 a block is 65,536 blocks, and
-- 2^25 is 131,072. The host dispatches the grid as a rectangle and the kernel reads its block count from its scalars
-- (every block past it returns), so the result equals the sequential one.
--
-- ==
-- entry: main
-- input { 16777216i64 } output { 50331645i64 }
-- input { 33554432i64 } output { 100663291i64 }

entry main (n: i64) : i64 =
  let xs = map (\i -> i % 7) (iota n)
  in i64.sum (map (\x -> x) xs)

-- The map kept as the output, so it is not fused into the reduction: its kernel is the one over the limit.
entry mapped (n: i64) : [n]i32 =
  map (\i -> i32.i64 (i % 7)) (iota n)
