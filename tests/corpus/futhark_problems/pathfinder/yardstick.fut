-- THE YARDSTICK: the published kernel of diku-dk/futhark-benchmarks
-- rodinia/pathfinder/pathfinder.fut (commit c9e2ccca, `published/`), as an
-- entry point on a given wall grid. The published `main` makes its own walls
-- (Sobol numbers modulo 10) from (cols, rows); this entry takes the walls as
-- its input. The helper `min` and the kernel (the copy of row 0 and the loop)
-- are the published text unchanged; `sobol_walls.fut` and the test
-- `the_yardstick_is_the_published_kernel` check it gives medium.out on the
-- published walls.

def min (a: i32, b: i32) : i32 = if (a <= b) then a else b

entry main [rows][cols] (wall: [rows][cols]i32) : [cols]i32 =
  let result = copy (wall[0])
  ---------------
  -- 1. Kernel --
  ---------------
  in loop (result) for t < (rows - 1) do
       map5 (\i prev res next wall' ->
               let res = if i > 0 then min (res, prev) else res
               let res = if i < cols - 1 then min (res, next) else res
               in wall' + res)
            (iota cols)
            (rotate (-1) result)
            result
            (rotate 1 result)
            wall[t + 1]
