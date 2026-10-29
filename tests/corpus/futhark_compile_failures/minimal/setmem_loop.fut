-- setmem_loop, minimal: a loop's memory parameter starts at one buffer and is rotated to another inside the kernel's bounds check (patch 0008).
entry main [n] (k: i64) (is: [n]i32) (vs: [n]i32) : [k]i32 =
  let zeros_k = replicate k 0i32
  in tabulate k (\j -> (loop s = hist (+) 0i32 k (map i64.i32 is) vs for _t < k - 1 do zeros_k)[j])
