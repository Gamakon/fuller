-- input: -g [100000]f32
-- A 1-D kingdom-shaped program: scan, map2, reverse and a scaling map.
entry main (xs: []f32) : []f32 = map (\x -> x * 3.0f32) (reverse (map2 (-) xs (scan (+) 0.0f32 xs)))
