-- THE TEST VIEWS OF THE PAIR CONSTRUCTS (docs/PLAN_futhark_kingdom.md, step 4c (b)): two small
-- views over two elements, sniffed from the two entries below, in which the kingdom must express
-- UltraDim's k-means constructs construct for construct
-- (lib/projection-voting-kmeans/src/gpu_kmeans.rs, `update_centroids` and `assign_points`):
--
-- * `kahan`: the Kahan-compensated mean of the values whose membership is 0 -- a sequential loop
--   over the points whose state is the PAIR (sum, compensation), each step guarded by the
--   membership, then the sum over the count (the count is UltraDim's third state; the kingdom's
--   pair holds two, so a program of the kingdom counts in a second loop: the same values);
-- * `argmin`: the index of the least distance -- the pair (best distance, best index) from the
--   first distance, over the rest in order, a later distance winning only when strictly smaller;
-- * `cell` (step 4c (c), master version 5): `update_centroids` for ONE (cluster, dimension) cell,
--   construct for construct -- a sequential loop over the points whose state is the TRIPLE (sum,
--   compensation, count), the membership test inside, each member's step `y = x - c; t = s + y;
--   c = (t - s) - y; s = t; count + 1`, then the sum over the count when the count is positive, else
--   the previous centre (`if (count > 0u)`: an empty cluster keeps its centre);
-- * `kernels` (4c (c)): the KERNEL BOUNDARY -- one kernel writes a buffer the next one reads; here a
--   `let` of an `[n]i32` computed once at the top (a flag a point) and a loop that reads it.
--
-- These entries are the yardsticks the kingdom's programs are compared with, bit for bit, through
-- `futhark c` (src/futhark_sig/tests_view.rs); they are spelled with Futhark's own tuple loops and
-- range loop so the comparison is between two spellings, not one.
entry kahan [n] (xs: [n]f32) (ms: [n]i32) : f32 =
  let (s, _, k) = loop (s, c, k) = (0f32, 0f32, 0i32) for i < n do
    if ms[i] == 0 then let y = xs[i] - c let t = s + y in (t, (t - s) - y, k + 1) else (s, c, k)
  in s / f32.i32 k

entry argmin [n] (ds: [n]f32) : i32 =
  let (_, b) = loop (best, b) = (ds[0], 0i32) for c in 1..<n do
    if ds[c] < best then (ds[c], i32.i64 c) else (best, b)
  in b

entry cell [n] (xs: [n]f32) (ms: [n]i32) (olds: [n]f32) : f32 =
  let (s, _, k) = loop (s, c, k) = (0f32, 0f32, 0i32) for i < n do
    if ms[i] == 0 then let y = xs[i] - c let t = s + y in (t, (t - s) - y, k + 1) else (s, c, k)
  in if k > 0 then s / f32.i32 k else olds[0]

entry kernels [n] (xs: [n]f32) : f32 =
  let flags = tabulate n (\i -> if xs[i] - 0.5 < 0 then 0i32 else 1i32)
  in loop s = 0f32 for i < n do if flags[i] == 1 then s + xs[i] else s
