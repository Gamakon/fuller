-- input: -g [64][64]f32 -g [64][64]f32 -g [64][64]f32 -g [64][64]f32 -g [64][64]f32 -g [64][64]f32 -g [64][64]f32 -g [64][64]f32 -g [64][64]f32
-- The A7 winner of docs/measurements/2026-10-27_futhark_shapes_A7.md, in the
-- form src/futhark_shapes.rs::program renders it, at n = m = p = 64.
def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =
  map (\a -> map (\b -> reduce (+) 0.0 (map2 (*) a b)) (transpose B)) A

def kernel [m][n][p] (m_n_n: [n][n]f32) (m_n_m: [n][m]f32) (m_n_p: [n][p]f32) (m_m_n: [m][n]f32) (m_m_m: [m][m]f32) (m_m_p: [m][p]f32) (m_p_n: [p][n]f32) (m_p_m: [p][m]f32) (m_p_p: [p][p]f32) : [n][p]f32 =
  map2 (\xs ys -> map2 (+) xs ys) (matmul (m_n_m) (m_m_p)) (matmul (m_n_m) (m_m_p))

entry main (m_n_n: [64][64]f32) (m_n_m: [64][64]f32) (m_n_p: [64][64]f32) (m_m_n: [64][64]f32) (m_m_m: [64][64]f32) (m_m_p: [64][64]f32) (m_p_n: [64][64]f32) (m_p_m: [64][64]f32) (m_p_p: [64][64]f32) : [64][64]f32 = kernel m_n_n m_n_m m_n_p m_m_n m_m_m m_m_p m_p_n m_p_m m_p_p
