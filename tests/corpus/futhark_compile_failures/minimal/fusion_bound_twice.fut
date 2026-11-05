-- fusion_bound_twice, minimal: a scan fused into a reduction is a stream, and a second scan of the same input fuses into it horizontally; the second scan's stream named its results after its own lambda's parameters (patch 0015). Fails in stock Futhark 0.28.1 too.
entry main [n] (xs: [n]i32) : (i32, [n]i32) = (i32.minimum (scan (+) 0 xs), scan i32.max i32.lowest xs)
