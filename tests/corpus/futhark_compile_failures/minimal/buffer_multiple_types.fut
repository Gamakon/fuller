-- buffer_multiple_types, minimal: a histogram with f32.max (no WGSL atomic) updates its buckets by compare-and-exchange of their bits (i32) and reads them as f32 in one kernel; the webgpu code generator gave a buffer one type (patch 0013).
entry main [n] (is: [n]i64) (xs: [n]f32) : []f32 = hist f32.max f32.lowest 10 is xs
