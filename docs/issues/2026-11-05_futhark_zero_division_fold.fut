-- Futhark 0.28.1: the simplifier folds `0 / x` to the zero dividend, where IEEE gives NaN for 0/0 and 0/NaN and
-- -0.0 for 0/-3. `futhark run` follows IEEE; `futhark c` does not. Whether the fold fires depends on whether the
-- simplifier sees the zero (`hidden`: a zero carried through a loop is not folded).
--
--   futhark c 2026-11-05_futhark_zero_division_fold.fut -o zf
--   echo '0f32'      | ./zf -e scalar    # 0f32            (IEEE: f32.nan)
--   echo '-3f32'     | ./zf -e scalar    # 0f32            (IEEE: -0f32)
--   echo 'f32.nan'   | ./zf -e scalar    # 0f32            (IEEE: f32.nan)
--   echo '[0f32, -3f32]' | ./zf -e array    # [0f32, 0f32]   (IEEE: [f32.nan, -0f32])
--   echo '1i64 [0f32, -3f32]' | ./zf -e hidden   # [f32.nan, -0f32] (IEEE, no fold)
--   echo '0f32'      | futhark run -e scalar 2026-11-05_futhark_zero_division_fold.fut   # f32.nan

entry scalar (x: f32) : f32 = 0f32 / x

entry array [m] (s: [m]f32) : [m]f32 = map2 (/) (replicate m 0f32) s

entry hidden [m] (n: i64) (s: [m]f32) : [m]f32 =
  let z = loop acc = replicate m 0f32 for _i < n do map (* 2f32) acc
  in map2 (/) z s
