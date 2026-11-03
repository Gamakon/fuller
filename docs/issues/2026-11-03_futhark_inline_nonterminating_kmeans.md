# Futhark's inliner does not terminate on `f32.min (0f32 / 0f32) 0f32` (webgpu branch, 0.29.0 prerelease)

*fuller, 2026-11-03. For the compiler team (diku-dk/futhark, the `webgpu` branch, and our rust-wgpu fork).*

## Summary

A legal program that Futhark 0.28.1 compiles in 0.2 s is never finished by the webgpu branch's compiler (0.29.0
prerelease at `14e9ccbf`, and our fork of it): the pass "Inline conservatively" (and "Inline aggressively" alike),
run after one simplify pass, does not terminate. Not a fusion fault: `--fuse=10` runs for ever too, and the pass log
(`-v`) stops at

```
[  +  0.002] Running pass: simplify
[  +  0.003] Running pass: Inline conservatively
```

The one-line reproducer (`docs/issues/futhark_inline_nonterminating.fut`):

```futhark
entry main (x: f32) : f32 = f32.min (0f32 / 0f32) 0f32
```

The first simplify folds `0f32 / 0f32` to `f32.nan` and leaves `apply min_5243(f32.nan, 0.0f32)` (a call of the
prelude's `min`, `fmin32`), and the inliner's own simplification of that call never settles. The branch's `fmin`
constant folding treats NaN as missing (`fmin x y | isNaN x = y | isNaN y = x`, `Language.Futhark.Primitive`), a
rule 0.28.1 does not have; which rewrite loops against it we have not traced. The variable form
`f32.min (0f32 / 0f32) x` compiles, and so does `futhark dev -e` alone, `futhark dev --inline-conservatively` alone
(no simplify first, so no NaN constant yet) and `futhark dev -e -e`.

## Versions, commands and outcomes

Each compile killed after 120 s (a normal batch of 5 programs compiles in 0.36 s on this path, one program in 0.2 s;
the kill guard is in the measurement script, the compiler ran at 99 % CPU until killed).

| compiler | command | outcome |
|---|---|---|
| Futhark 0.28.1 (Homebrew) | `futhark c --library` | compiled, 0.21 s |
| Futhark 0.28.1 | `futhark dev <the sequential pass list> --fuse=10 --backend=c --library` (fuller's bounded-fusion path) | compiled, 0.22 s |
| fork `rust-wgpu @ 630db2b` (webgpu branch `14e9ccbf` + 11 patches) | `futhark rust-wgpu` | killed at 120 s |
| the same fork | `futhark c --library` | killed at 120 s |
| the same fork | `futhark rust-wgpu --fuse=10` (patch 0012) | killed at 90 s |
| the same fork | `futhark dev -s` (the standard pipeline) | killed at 90 s |
| the same fork | `futhark dev --gpu` | killed at 90 s |
| the same fork | `futhark dev -e --inline-conservatively` / `-e --inline-aggressively` | killed at 15 s |
| the same fork | `futhark dev -e`, `dev --inline-conservatively`, `dev -e -e` | compiled |

The run's program is in the compile-failure corpus as
`tests/corpus/futhark_compile_failures/inline_nonterminating__f1b56069eb2f.fut` (the k-means candidate as the runner
emitted it: `def k7_main` and its range entry). The batch it stalled the run in, 13 programs, is
`docs/issues/futhark_fusion_nonterminating_batch.fut` (named before the pass was known).

## The value, on the compiler that finishes

The program computes a value on every stored k-means case (`tests/corpus/futhark_problems/kmeans/*.in`): Futhark
0.28.1's `futhark c` and its `dev --fuse=10` executable give byte-identical text output on all 7 cases
(`held_n1000_k32`, `held_n500_k16`, `test_n1000_k32`, `train_n1000_k16`, `train_n1000_k50`, `train_n200_k4`,
`train_n500_k8`; the timing input has another type and was not run), and Futhark's interpreter (`futhark run`)
agrees with them to its printed precision (it prints 8 significant digits, the compiled program 9: largest difference
2.9e-8 over 49,152 values of `held_n1000_k32`, none beyond the last printed digit).

## What distinguishes the failing program from one that compiles

The k-means candidate `k7` differs from its closest sibling `k0` (ratio 0.85 by word) in a few mutated cells; the one
construct no compiling sibling has is

```
f32.min ((((0f32, 0f32, 0f32)).1) / (((0f32, 0f32, 0f32)).0)) (((0f32, 0f32)).0)
```

a tuple-projected `0 / 0` as the first operand of `f32.min` against a constant `0`: the one-line reproducer above.

## How found

The k-means evolution run of 2026-11-03 (`/private/tmp/claude-501/fbench/kmeans_qqs_s1`) stalled at generation 8: its
`futhark rust-wgpu` compile of a 13-program batch (`rust_build/b69/prog.fut`) ran 19 minutes at 99 % CPU before it was
killed, and the retry of the batch stalled the same way (the runner was stopped at 17:05Z after its hour, at
generation 7 of the checkpoint). The batch was bisected a program at a time on the CPU with a 120 s kill: 11 of the
13 compile in 0.21 to 0.32 s, `k2` fails at once with a known limitation, `k7` is the one that never finishes. A
sibling run's compile (`fable_m3/m3/new_pathfinder_R_s3/rust_build/b62/prog.fut`, the pathfinder problem) had been
running for 5 hours 34 minutes at 99 % CPU at the time of the bisection, so the fault is not k-means's alone.

The batch compile had `n` times the per-candidate path's 120 s, 26 minutes for the batch of 13; it now has a limit
derived from the measured compile times on the Futhark stage (`futhark_batch::COMPILE_LIMIT_FLOOR`, 30 s, or 30 times
the median of the host's finished Futhark stages; rustc keeps its own bound), the stuck batch is split as a failed compile is, its source saved under the run's `bugs/`,
and a single program that did not finish is tried once more with `--fuse=10` (the fusion bound: it does rescue the
program of `docs/RESCUE_futhark.md`, whose fusion never settles, 0.17 s against a kill at 53 s; it does not rescue
this one).

## A second fault found on the way: the threaded runtime livelocks at 16 capabilities

While testing the bound, the fork sometimes did not finish a program it compiles in 0.3 s: 6 times in about 5,800
compiles (once in a 1,890-compile sweep, once in 5 of 6 runs of a 758-compile test), never the same program twice,
never in 290 shell repeats of those programs. The fork is built with GHC 9.14.1's threaded runtime and
`-with-rtsopts=-maxN16 -qg1 -A32M`. A 5 s stack sample (`sample`) of one stuck compiler: the main Haskell thread in
`stg_BLACKHOLE_info` under `schedule` in every sample, all 40 worker threads idle (`kevent`, `pthread_cond_wait`):
the runtime is livelocked on a thunk another capability claimed, not the compiler's passes. With `+RTS -N1` the same
test ran twice (1,516 compiles) with no hang and at the same speed (120 s a run), and every corpus program's generated
Rust is byte-identical to the 16-capability compiler's (377 of 377 that compile). fuller now runs the fork with
`+RTS -N1 -RTS` on every compile (`futhark_rust::COMPILER_RTS`), and the batch time limit stays as the backstop. For
the GHC/Futhark side: a 16-capability threaded build livelocking in `stg_BLACKHOLE_info` on aarch64-darwin, rate
about 1 in 1,000 compiles when 4 to 12 compiles run at once; we have the sample, not a deterministic reproducer.

## What we need

A fix of the inliner's simplification (or of the `fmin`/`fmax` NaN folding it loops against), so that a legal
program is compiled or rejected with a message; a compiler that runs for ever cannot be told from our own bug by a
running search. Until then our workaround is the compile limit and the batch split; we will add the fix to our patch
series (`futhark-rust-backend/patches`) when it exists.

## In our test suite

The run's program is in the compile-failure corpus as `inline_nonterminating__f1b56069eb2f.fut`, marked in
`known_faults.tsv` beside it with this issue's path and the error text `no answer within` (the compile is killed at
60 s there, `futhark_rust::corpus_compile_limit`). Both corpus tests (the rust path's and the webgpu path's) expect
it to fail that way and report it as known; the day the fork compiles it, the test fails and says to remove the
mark. The batch split on a compile that does not finish is proved by a mock compiler that sleeps
(`futhark_batch::tests::a_compile_that_does_not_finish_is_killed_at_the_limit_and_the_batch_is_split`).
