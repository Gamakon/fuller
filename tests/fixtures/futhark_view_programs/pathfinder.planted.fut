entry main [rows][cols] (walls: [rows][cols]i32) : [cols]i32 =
  let zeros_rows = replicate rows 0i32
  let zeros_cols = replicate cols 0i32
  in map2 i32.min (loop (s1: [cols]i32) = ((walls)[(0i64)]) for t1 < rows - 1 do let i1 = t1 + 1 in (tabulate cols (\j2 -> ((((walls)[(i1)])[(j2)]) + (i32.min (i32.min ((s1)[(j2)]) ((s1)[(i64.max 0 ((j2) - 1))])) ((s1)[(i64.min (cols - 1) ((j2) + 1))])))))) (loop (s1: [cols]i32) = ((walls)[(0i64)]) for t1 < rows - 1 do let i1 = t1 + 1 in (tabulate cols (\j2 -> ((((walls)[(i1)])[(j2)]) + (i32.min ((s1)[(j2)]) (i32.min ((s1)[(i64.max 0 ((j2) - 1))]) ((s1)[(i64.min (cols - 1) ((j2) + 1))])))))))
