-- THE QUANTISER of the stored cases' inputs: `futhark dataset`'s uniforms in [0, 1] (arbitrary 24-bit mantissas)
-- become multiples of 2^-6 in [0, 1), 64 levels, floor(u * 64) / 64 with 64/64 folded to 63/64. Every operation is
-- exact in f32 (a scaling by a power of two, a floor, a minimum), so the quantised value is a function of the drawn
-- value alone. Why: a k-means step sums squared differences over d and sums points over a cluster, and the GPU adds
-- in another order than `futhark c` and the host interpreter; with every input a multiple of 2^-6 every partial
-- sum any order can form is exact in f32 (`src/futhark_sig/tests_kmeans.rs`, the dyadic test, holds the bounds), so
-- the order is irrelevant and the step's output is one set of bits on every path (up to the final divisions).
entry main [n] [d] [k] (points: [n][d]f32) (centres: [k][d]f32) : ([n][d]f32, [k][d]f32) =
  let q (x: f32) : f32 = f32.min 63 (f32.floor (x * 64)) / 64
  in (map (map q) points, map (map q) centres)
