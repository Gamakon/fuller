-- The published benchmark's own walls (its initialisation, unchanged), as an
-- entry point, so the yardstick can be run on them and compared with
-- published/medium.out.
import "published/pathfinder"

entry main (cols: i64) (rows: i64) : [rows][cols]i32 =
  let dirVct = sobolDirVcts ()
  let wall_flat =
    map (\i ->
           sobolInd (dirVct, i + 1) % 10)
        (iota (rows * cols))
  in unflatten wall_flat
