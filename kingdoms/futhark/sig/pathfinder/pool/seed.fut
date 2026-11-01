-- Pathfinder: shift the running cost row left and right, then combine the
-- three candidate neighbours with map2 i32.min and add the wall row with map2 (+).

def shl [cols] (s: [cols]i32) : [cols]i32 =
  tabulate cols (\j -> s[i64.max 0 (j - 1)])

def shr [cols] (s: [cols]i32) : [cols]i32 =
  tabulate cols (\j -> s[i64.min (cols - 1) (j + 1)])

entry main [rows][cols] (walls: [rows][cols]i32) : [cols]i32 =
  loop (s: [cols]i32) = walls[0] for t < rows - 1 do
    let i = t + 1
    in map2 (+) walls[i] (map2 i32.min s (map2 i32.min (shl s) (shr s)))
