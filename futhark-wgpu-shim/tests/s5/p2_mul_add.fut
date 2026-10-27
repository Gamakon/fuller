-- input: -g [100000]f32 -g [100000]f32
-- A true multiply-add: two roundings in C, one if the GPU fuses it into fma.
entry main (xs: []f32) (ys: []f32) : []f32 = map2 (\x y -> x * y + 1.0f32) xs ys
