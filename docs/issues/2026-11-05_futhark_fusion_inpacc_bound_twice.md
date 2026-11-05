# Futhark fusion: "Name inpacc_... bound twice" on histogram and pathfinder rows (2026-11-05)

**What happens.** `futhark c --library` (Futhark 0.28.1, Homebrew, the C backend of our evaluator) stops on random
histogram and pathfinder rows of 225 to 1,100 nodes:

    Internal compiler error.  Please report this:
      https://github.com/diku-dk/futhark/issues
    Type error after pass 'Fuse SOACs':
    In function kq3_11066
    ...
    Name inpacc_59768 bound twice

**Where.** The size-bound measurement of plan step 5a (`/private/tmp/claude-501/fbench/plan5a/size_bound/`,
2026-11-04): 11 of the 15 rows in `out_small2/compile_failures/` (1 to 4 a batch of 25 to 28 rows; the other 4 were
killed at the compile limit on both compilers, another fault). The reproducer beside this note,
`2026-11-05_futhark_fusion_inpacc_bound_twice.fut`, is the batch `ice/histogram_225_275_k04.fut` (4 programs; `kq3`
fails alone).

**Which compilers.** Futhark 0.28.1 (`futhark c`, `futhark opencl`), our fork (`futhark c`, `futhark rust-wgpu`,
`futhark webgpu`, before patch 0015) and Futhark's master alike: the fault is upstream's, in the standard pipeline every
backend runs. `src/Futhark/Optimise/Fusion/TryFusion.hs` and `src/Futhark/Analysis/HORep/SOAC.hs` are the same on
master (2026-11-05) as in the fork's base. No upstream issue names it.

## Reduced

    entry main [n] (xs: [n]i32) : (i32, [n]i32) =
      (i32.minimum (scan (+) 0 xs), scan i32.max i32.lowest xs)

(`tests/corpus/futhark_compile_failures/minimal/fusion_bound_twice.fut`). `(i32.minimum (scan (+) 0 xs), scan (+) 0 xs)`
fails the same way.

## Cause

The first scan fuses with the reduction into a stream (`soacToStream` turns the scan producer into a stream). The
second scan of the same input then fuses into that stream horizontally: the producer is a stream and the kernel is
not, so `fuseSOACwithKer` turns the KERNEL into a stream too (the `(_, SOAC.Stream {}, _)` case). For a scan,
`soacToStream` returns the stream's extra accumulator results named after the stream lambda's own parameters
(`map paramIdent inpacc_ids`), and the fusion makes those names output names of the kernel. A producer's lambda is
renamed on its way into the fused SOAC (`renameLambda` at the top of `fuseSOACwithKer`); the kernel's is not. So the
statement binds `inpacc_N` and the lambda inside it binds `inpacc_N`: bound twice, and the type checker after the pass
stops. (The case that turns a producer into a stream is safe: the producer's lambda is renamed.)

## Fix

Patch 0015 (`futhark-rust-backend/patches/0015-fix-fusion-a-kernel-turned-into-a-stream-has-its-lam.patch`): in that
case the kernel's stream gets its lambda renamed before the recursive fusion. Sound: renaming a lambda gives every name
it binds a fresh name consistently and leaves its free variables alone, so the program means the same; the result names
are then bound once. Only a scan's stream is renamed (a reduction's returns no such names), and a failed fusion attempt
rolls its fresh names back (`TryFusion` is a `StateT VNameSource Maybe`), so a program that compiled before compiles
to the same code: measured on 582 programs, byte-identical (README, rebuild of 2026-11-05 03:15Z). Futhark test: `tests/fusion/scan-stream-horizontal.fut` in the fork (passes on `c`; 0 of 1
passes on 0.28.1).

**Checked (CPU only).** With the rebuilt fork: the reproducer, the reduced programs and all 11 failing rows compile
with `futhark c --library` (the 4 other rows are killed at 300 s on both compilers, another fault, not this one), and
the reproducer and reduced programs with `futhark rust-wgpu` and `futhark webgpu`; the fork's test passes on `c`. The
reproducer is in the compile-failure corpus as `tests/corpus/futhark_compile_failures/fusion_bound_twice__86b12227050e.fut`
(index kind `fusion_bound_twice`) with the reduced program in `minimal/`; the corpus tests require both to compile
through the rust path and the webgpu path. Not in `known_faults.tsv`.

## The C backend

**The C backend does NOT use the fork.** fuller's C path (`futhark_exec`, `Command::new("futhark")`, and the
size-bound script's `futhark c --library`) runs the `futhark` on the PATH, `/opt/homebrew/bin/futhark`, Futhark 0.28.1
unpatched. So these rows still fail on the C backend; they compile on the rust-wgpu and webgpu paths.

**Decision: the C backend stays on Futhark 0.28.1.** Pointing it at the fork would fix these rows, but the fork is
the webgpu branch (0.29.0 prerelease), a different compiler from the 0.28.1 every C-backend measurement and claim was
made with (the C interpreter's bit-identity, the A7 results, the fusion bound's pass list, which mirrors 0.28.1's
`standardPipeline`), and it brings a known regression: the k-means program of
`docs/issues/2026-11-03_futhark_inline_nonterminating_kmeans.md` never finishes on the webgpu branch and compiles in
0.21 s on 0.28.1. The consequence of staying: a row that hits this fault is a compile failure on the C backend until
Futhark fixes it upstream (this note, below, is the report), about 1 to 4 in 25 rows of 225 to 1,100 nodes.

## For the Futhark team

Futhark 0.28.1 and master: the program under "Reduced" fails with `Type error after pass 'Fuse SOACs': Name inpacc_N
bound twice`. In `fuseSOACwithKer`'s `(_, SOAC.Stream {}, _)` case the kernel is turned into a stream with
`soacToStream`, whose scan case names the new accumulator results after the stream lambda's parameters; unlike the
producer, the kernel's lambda is not renamed, so those names are bound both by the statement and by the lambda.
Renaming the kernel stream's lambda (`renameLambda`) before the recursive `fuseSOACwithKer` fixes it; the patch and a
test (`tests/fusion/scan-stream-horizontal.fut`) are in `futhark-rust-backend/patches/0015-*`.
