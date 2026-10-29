-- simplify_type_error, minimal: two hists with different bin counts in one map; the flatten pass gave the second the first's destination shape (patch 0009). Fails in stock Futhark 0.28.1's GPU pipeline too.
entry main [n] (k: i64) (is: [n]i64) (vs: [n]i32) (ws: [n]i32): ([k][k]i32, [k][n]i32) =
  unzip (tabulate k (\_ -> (hist (+) 0 k is vs, hist (+) 0 n is ws)))
