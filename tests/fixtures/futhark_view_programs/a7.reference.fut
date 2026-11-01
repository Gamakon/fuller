def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =
  map (\a -> map (\b -> reduce (+) 0.0 (map2 (*) a b)) (transpose B)) A

-- the kernel: shape-polymorphic over the interface's roles
def kernel [m][n][p] (m_n_n: [n][n]f32) (m_n_m: [n][m]f32) (m_n_p: [n][p]f32) (m_m_n: [m][n]f32) (m_m_m: [m][m]f32) (m_m_p: [m][p]f32) (m_p_n: [p][n]f32) (m_p_m: [p][m]f32) (m_p_p: [p][p]f32) : [n][p]f32 =
  map2 (\xs ys -> map2 (+) xs ys) (matmul (m_n_m) (m_m_p)) (transpose (matmul (transpose (m_m_p)) (transpose (m_n_m))))

-- the entry point: concrete sizes
entry main (m_n_n: [2][2]f32) (m_n_m: [2][3]f32) (m_n_p: [2][4]f32) (m_m_n: [3][2]f32) (m_m_m: [3][3]f32) (m_m_p: [3][4]f32) (m_p_n: [4][2]f32) (m_p_m: [4][3]f32) (m_p_p: [4][4]f32) : [2][4]f32 = kernel m_n_n m_n_m m_n_p m_m_n m_m_m m_m_p m_p_n m_p_m m_p_p
