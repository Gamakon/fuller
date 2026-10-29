entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : [i64.i32 k]i32 =
  let k = i64.i32 k
  in tabulate k (\b ->
       i32.sum (tabulate n (\i ->
         let d = i32.abs ((is[i] % i32.i64 k) - i32.i64 b)
         in if (d - 1) < 0i32 then vs[i] else 0)))
