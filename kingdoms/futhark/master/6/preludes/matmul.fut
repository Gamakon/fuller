def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =
  map (\a -> map (\b -> reduce (+) 0.0 (map2 (*) a b)) (transpose B)) A
