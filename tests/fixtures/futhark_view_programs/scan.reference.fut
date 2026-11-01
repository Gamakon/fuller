entry main [n] (xs: [n]i32) : [n]i32 =
  let zeros_n = replicate n 0i32
  in scan (+) 0i32 (opaque (xs))
