-- input: -g [100000]f32
-- An f32 prefix sum: the parallel scan adds in another order than C's loop.
entry main (xs: []f32) : []f32 = scan (+) 0.0f32 xs
