# Futhark simplifier: `0 / x` is folded to `0`, which IEEE floats do not allow (2026-11-05)

**What happens.** Futhark 0.28.1's simplifier rewrites an `f32` division whose dividend it can see is a zero into that
zero. For IEEE floats that is wrong for three kinds of divisor: `0 / 0` and `0 / NaN` are NaN, and `0 / -3` is
`-0.0`. `futhark run` (the reference interpreter) gives the IEEE values; `futhark c` gives `0`. Measured on Futhark
0.28.1 (GHC 9.14.1), macOS, with the reproducer beside this note, `2026-11-05_futhark_zero_division_fold.fut`:

| entry, input            | `futhark c`        | `futhark run` (IEEE)  |
|-------------------------|--------------------|-----------------------|
| `scalar 0f32`           | `0f32`             | `f32.nan`             |
| `scalar -3f32`          | `0f32`             | `-0.0f32`             |
| `scalar f32.nan`        | `0f32`             | `f32.nan`             |
| `array [0f32, -3f32]`   | `[0f32, 0f32]`     | `[f32.nan, -0.0f32]`  |
| `hidden 1i64 [0f32, -3f32]` | `[f32.nan, -0f32]` | `[f32.nan, -0.0f32]` |

**It is the simplifier.** `futhark dev -s` (one simplification pass) turns `entry scalar (x: f32) = 0f32 / x` into
the constant `0.0f32`.

**It depends on what the simplifier can see.** The fold fires on a literal zero, on `replicate m 0f32`, on
`tabulate m (\_ -> 0f32)`, on an element of such an array and on `map (\_ -> 0.0) xs` fused into the division. It does
not fire when the zero is carried through a loop (`hidden`), so the same division gives two results in two programs
depending on fusion and inlining. A program's numbers should not depend on that.

**What is not folded.** `0f32 * x` is not: `0 * NaN` and `0 * inf` give NaN, as IEEE says. A dividend of `-0f32`
folds to `-0.0` (`(-0f32) / 0f32` gives `-0f32`, IEEE NaN), so the rule appears to return the dividend itself.

**Suggested fix.** Fold `0 / x` to `0` only for integer types, or for floats only where `x` is known to be finite and
positive. The same rule for `x / 1` is sound and can stay.

**How we work around it.** The Futhark kingdom's interpreter (`src/futhark_sig/interp6.rs`) keeps IEEE's division, and
the comparison of the interpreter with the `c` backend (`src/futhark_sig/tests_interp6.rs`, `zero_fold`) runs the
program a second time with every zero dividend folded. A cell of the `c` output may differ from the interpreter's only
where the two runs differ (a cell a zero dividend reached) and must then equal the folded run's. Any other difference,
including a NaN of another origin that `c` turned into `0`, fails the test.

Not yet reported upstream; this note is for the Futhark team.
