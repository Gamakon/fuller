-- input: -g [100000]i32
-- An i32 sum: wrapping integer addition is associative, so any order agrees.
entry main (xs: []i32) : i32 = reduce (+) 0i32 (map (\x -> x * 3i32) xs)
