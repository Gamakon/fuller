-- THE YARDSTICK: the published program of diku-dk/futhark-benchmarks
-- micro/reduce.fut (commit c9e2ccca, `published/`), its entry
--   entry sum_i32 = i32.sum
-- as `main` with the declared signature `xs: [n]i32 -> i32`. The body is the
-- published text unchanged; the test
-- `the_published_micro_programs_and_yardsticks_give_the_stored_outputs`
-- checks the published entry gives every stored output.

entry main [n] (xs: [n]i32) : i32 = i32.sum xs
