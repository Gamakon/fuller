-- THE F64 REFERENCE OF ULTRADIM'S K-MEANS STEP: the prototype's three kernels (qdrant lib/projection-voting-kmeans/
-- src/gpu_kmeans.rs: compute_distances, assign_points, update_centroids) in f64, construct for construct:
--   * each point's distance to each centre is the in-order sum of squared differences, then sqrt;
--   * each point's centre is the nearest, the FIRST on a tie (`distance < best_distance` replaces only a strictly
--     smaller one);
--   * each (cluster, coordinate) is the Kahan-compensated sum over the points in point order
--     (y = x - c; t = s + y; c = (t - s) - y; s = t) divided by the count;
--   * a cluster with no point keeps its centre (`if (count > 0u)`).
-- It is A CHECK OF THE TRANSLATION of those kernels into a gene (docs/PLAN_futhark_kingdom.md 4c (d)), the same
-- algorithm as futhark_sig::kmeans_converged::ultradim_step_f64 (a test requires the two equal bit for bit on every
-- stored case: both are sequential f64). IT NEVER SCORES: k-means has no exact answer, and the fitness runs a program
-- to UltraDim's stopping rule and measures the clustering (docs/KMEANS_FITNESS.md).
--   futhark c reference64.fut -o r64; ./r64 -b < case.in    -- ([n]i32 assignments, [k][d]f64 centres)
def nearest [k] [d] (centres: [k][d]f64) (p: [d]f64) : i32 =
  let dist (c: [d]f64) : f64 = f64.sqrt (loop s = 0.0 for t < d do s + (p[t] - c[t]) * (p[t] - c[t]))
  let (best, _) =
    loop (best, best_distance) = (0i32, dist centres[0]) for j in 1..<k do
      let x = dist centres[j]
      in if x < best_distance then (i32.i64 j, x) else (best, best_distance)
  in best

def kahan_mean [n] (xs: [n]f64) (mine: [n]bool) (old: f64) : f64 =
  let (s, _, count) =
    loop (s, c, count) = (0.0, 0.0, 0i64) for i < n do
      if mine[i] then
        let y = xs[i] - c
        let t = s + y
        in (t, (t - s) - y, count + 1)
      else (s, c, count)
  in if count > 0 then s / f64.i64 count else old

entry main [n] [d] [k] (points: [n][d]f32) (centres: [k][d]f32) : ([n]i32, [k][d]f64) =
  let p = map (map f64.f32) points
  let c = map (map f64.f32) centres
  let assignments = map (nearest c) p
  let new_centres =
    tabulate k (\j ->
      let mine = map (\a -> a == i32.i64 j) assignments
      in tabulate d (\t -> kahan_mean (map (\row -> row[t]) p) mine c[j, t]))
  in (assignments, new_centres)
