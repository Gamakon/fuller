-- THE YARDSTICK of A7: the product of two matrices computed TWICE, once directly and once the transposed way, and
-- the two added (docs/PLAN_futhark_product.md; futhark-wgpu-shim/tests/s5/a7_reference.fut is the same kernel at
-- n = m = p = 64). The kernel is the text src/futhark.rs::program renders for the reference gene; the entry has the
-- interface's declared signature (the nine matrices, one a class, in the view's class order) polymorphic in the
-- roles, so it runs on every stored case.

def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =
  map (\a -> map (\b -> reduce (+) 0.0 (map2 (*) a b)) (transpose B)) A

def kernel [m][n][p] (m_n_n: [n][n]f32) (m_n_m: [n][m]f32) (m_n_p: [n][p]f32) (m_m_n: [m][n]f32) (m_m_m: [m][m]f32) (m_m_p: [m][p]f32) (m_p_n: [p][n]f32) (m_p_m: [p][m]f32) (m_p_p: [p][p]f32) : [n][p]f32 =
  map2 (\xs ys -> map2 (+) xs ys) (matmul (m_n_m) (m_m_p)) (transpose (matmul (transpose (m_m_p)) (transpose (m_n_m))))

entry main [n][m][p] (m_n_n: [n][n]f32) (m_n_m: [n][m]f32) (m_n_p: [n][p]f32) (m_m_n: [m][n]f32) (m_m_m: [m][m]f32) (m_m_p: [m][p]f32) (m_p_n: [p][n]f32) (m_p_m: [p][m]f32) (m_p_p: [p][p]f32) : [n][p]f32 =
  kernel m_n_n m_n_m m_n_p m_m_n m_m_m m_m_p m_p_n m_p_m m_p_p
