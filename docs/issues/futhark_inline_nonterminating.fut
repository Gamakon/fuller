-- The Futhark webgpu branch (0.29.0 prerelease at 14e9ccbf, and our rust-wgpu fork of it) never finishes compiling
-- this program: `futhark c`, `futhark rust-wgpu`, `futhark dev -e --inline-conservatively` and
-- `futhark dev -e --inline-aggressively` all run for ever in the inlining pass that follows the first simplify
-- (which folds `0f32 / 0f32` to `f32.nan`). Futhark 0.28.1 compiles it in 0.2 s. `f32.min (0f32 / 0f32) x` (a
-- variable in place of the constant) compiles; so does `dev -e` alone, `dev --inline-conservatively` alone, and
-- `dev -e -e`. See docs/issues/2026-11-03_futhark_inline_nonterminating_kmeans.md.
entry main (x: f32) : f32 = f32.min (0f32 / 0f32) 0f32
