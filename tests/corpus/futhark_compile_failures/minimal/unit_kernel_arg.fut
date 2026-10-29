-- unit_kernel_arg, minimal: a unit scalar reaches a kernel (hist's dest initialisation); rust-wgpu had no unit kernel argument (patch 0007).
entry main [n] (k: i64) (is: [n]i64) (vs: [n]i32) : [k]i32 =
  let zeros_k = replicate k 0i32
  in tabulate k (\j -> zeros_k[j] + (hist (+) 0 k is vs)[j])
