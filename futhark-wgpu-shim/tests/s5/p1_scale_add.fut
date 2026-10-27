-- input: -g [100000]f32 -g [100000]f32
-- Exact arithmetic: 2*x is exact, so one rounding per element on any device.
entry main (xs: []f32) (ys: []f32) : []f32 = map2 (\x y -> x * 2.0f32 + y) xs ys
