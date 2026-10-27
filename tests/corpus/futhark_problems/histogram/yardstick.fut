-- THE YARDSTICK: the published program of diku-dk/futhark-benchmarks
-- micro/reduce_by_index.fut (commit c9e2ccca, `published/`), its entry
-- sum_i32 and the helper `bins`, as `main`: the sum of the values `vs` in
-- each of `k` bins, a value's bin its index `is` modulo `k`. The text is the
-- published text unchanged but for the entry's name; the test
-- `the_published_micro_programs_and_yardsticks_give_the_stored_outputs`
-- checks the published entry gives every stored output.

def bins k is = map (% k) is |> map i64.i32

entry main [n] (k: i32) (is: [n]i32) (vs: [n]i32) : []i32 =
  hist (+) 0 (i64.i32 k) (bins k is) vs
