entry main [n] (xs: [n]i32) : i32 =
  let zeros_n = replicate n 0i32
  in i32.sum (xs)
