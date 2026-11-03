-- THE SHAPES of the magnitude cases (make_cases.sh, docs/PLAN_futhark_kingdom.md 4c item 3): futhark dataset's random
-- draws (never quantised) given a magnitude pattern. Every operation is a multiplication by a power of ten, so the
-- values stay random f32 of 24-bit mantissas.
--   mixed: coordinate t of every point and centre is scaled by 10^((t mod 7) - 3), so one case spans 10^-3 to 10^3
--          across its coordinates (d = 7 covers every scale once);
--   breaking: the FIRST `big` points are scaled by 10^4 among points around 1, so a cluster that takes them sums a
--          few values around 10^4 first and then many around 1: every later addition of an uncompensated f32 sum
--          rounds at the big sum's ulp (about 2^-13 at 2 x 10^4), a Kahan-compensated sum keeps the lost part (the
--          case built to separate the two; measured 2026-11-03 with the big points LAST the plain sum was as good as
--          Kahan's, 8.7e-7 against 1.0e-6, because only the four last additions rounded coarsely).
def pow10 (e: i64) : f32 = if e >= 0 then f32.i64 (10 ** e) else 1 / f32.i64 (10 ** (-e))

entry mixed [n] [d] [k] (points: [n][d]f32) (centres: [k][d]f32) : ([n][d]f32, [k][d]f32) =
  let scale (row: [d]f32) = map2 (\t x -> x * pow10 ((t % 7) - 3)) (iota d) row
  in (map scale points, map scale centres)

entry breaking [n] [d] [k] (big: i64) (points: [n][d]f32) (centres: [k][d]f32) : ([n][d]f32, [k][d]f32) =
  (map2 (\i row -> if i < big then map (* 10000) row else row) (iota n) points, centres)
