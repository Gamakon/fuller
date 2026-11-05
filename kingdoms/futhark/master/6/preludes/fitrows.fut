def fitrows [m][c] 't (l: i64) (z: t) (xs: [m][c]t) : [l][c]t =
  tabulate l (\i -> if i < m then xs[i] else replicate c z)
