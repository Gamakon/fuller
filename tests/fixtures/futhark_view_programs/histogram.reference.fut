entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : [i64.i32 k]i32 =
  let k = i64.i32 k
  let zeros_k = replicate k 0i32
  let zeros_n = replicate n 0i32
  in hist (+) 0i32 k (map i64.i32 (tabulate n (\j1 -> (((is)[(j1)]) % i32.i64 k)))) (vs)
