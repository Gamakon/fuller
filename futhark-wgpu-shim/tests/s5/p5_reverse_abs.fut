-- input: -g [100000]f32
-- Data movement and exact operations only.
entry main (xs: []f32) : []f32 = map (\x -> f32.abs (-x)) (reverse xs)
