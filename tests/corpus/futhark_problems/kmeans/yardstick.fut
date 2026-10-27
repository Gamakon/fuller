-- THE YARDSTICK: ONE STEP of the published program of diku-dk/futhark-benchmarks
-- rodinia/kmeans/kmeans.fut (commit c9e2ccca, `published/`). The published
-- `main` iterates to convergence from the first k points; one step of its
-- loop is: assign each point to its nearest centre (`find_nearest_point`),
-- then the new centres are the means of their points (`centroids_of`; an
-- empty cluster's centre is the zero vector, as published). The four
-- functions below the separator are the published text unchanged; `main`
-- applies them once to given centres.
-- ----
def euclid_dist_2 [d] (pt1: [d]f32) (pt2: [d]f32) : f32 =
  f32.sum (map (\x -> x * x) (map2 (-) pt1 pt2))

def closest_point (p1: (i32, f32)) (p2: (i32, f32)) : (i32, f32) =
  if p1.1 < p2.1 then p1 else p2

def find_nearest_point [k] [d] (pts: [k][d]f32) (pt: [d]f32) : i32 =
  let (i, _) =
    foldl (\acc (i, p) -> closest_point acc (i32.i64 i, euclid_dist_2 pt p))
          (0, f32.inf)
          (zip (indices pts) pts)
  in i

def centroids_of [n] [d] (k: i64) (points: [n][d]f32) (membership: [n]i32) : [k][d]f32 =
  let points_in_clusters =
    hist (+) 0 k (map i64.i32 membership) (replicate n 1)
  let cluster_sums =
    hist (map2 (+))
         (replicate d 0)
         k
         (map i64.i32 membership)
         points
  in map2 (\point n -> map (/ r32 (if n == 0 then 1 else n)) point)
          cluster_sums
          points_in_clusters

entry main [n] [d] [k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  centroids_of k points (map (find_nearest_point centres) points)
