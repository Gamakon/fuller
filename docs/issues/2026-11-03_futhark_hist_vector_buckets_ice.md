# Futhark internal compiler error: a `hist` with vector buckets inside a `tabulate` (GPU backends)

*fuller, 2026-11-03. For the compiler team (diku-dk/futhark, the `webgpu` branch, and our rust-wgpu fork).*

## Summary

A program that `futhark c` compiles and runs, and that `futhark dev --gpu` (the GPU IR pipeline) accepts, makes the
GPU code generator crash with an internal compiler error on both the upstream OpenCL backend and our rust-wgpu backend:

```
Internal compiler error (unhandled IO exception).
copyDWIMDest: array-typed target and prim-typed source hist_ne_12821 with slice
  [sext_i32_i64 (local_subhisto_i_13224), squot64 (...) (d_9932), sub_nw64 (...) (mul_nw64 (...) (d_9932)), nest_i_13227]
```

The construct: a generalised histogram whose buckets are vectors, `hist (map2 (+)) (replicate d 0f32) k is rows`
(operator `map2 (+)` on `[d]f32`, neutral element `replicate d 0f32`), evaluated INSIDE a `tabulate k (\i_k -> tabulate d
(\i_d -> ... (hist ...)[i_k][i_d] ...))`. The same `hist` at the top level of the program (the reference program of our
k-means problem) compiles and runs on every backend. The crash is in the GPU implementation of the histogram, where the
local sub-histogram is initialised from the neutral element: `hist_ne` is treated as a scalar (`prim-typed source`)
while the target slice is an array (one bucket of `d` floats).

## Reproducer

`docs/issues/futhark_hist_vector_buckets_ice.fut` (2.5 KB; one `def` and one `entry`, written by our evolution runner;
the `def` is one candidate program of the k-means problem, a mutation of UltraDim's centroid update). It compiles with
`futhark c` and fails on the GPU backends:

```
$ futhark --version
Futhark 0.29.0 (prerelease)   git: rust-wgpu @ 630db2b   (diku-dk/futhark webgpu branch at 14e9ccbf3a47c06ed585d15e6827d186dd74d8a0
                                                          plus our 11 patches, none of which touch the histogram code generator)
$ futhark c      futhark_hist_vector_buckets_ice.fut -o x     # ok
$ futhark dev --gpu futhark_hist_vector_buckets_ice.fut       # ok (the GPU pipeline's IR)
$ futhark opencl futhark_hist_vector_buckets_ice.fut -o x     # Internal compiler error ... copyDWIMDest ... hist_ne
$ futhark rust-wgpu futhark_hist_vector_buckets_ice.fut -o x  # the same error (our backend shares ImpGen's GPU histogram)
```

The `opencl` backend is upstream code in our fork, so the fault is upstream's `Futhark.CodeGen.ImpGen.GPU` histogram
implementation (the segmented histogram's local sub-histogram initialisation under an outer `tabulate`), not our patches.

## What distinguishes the failing program from one that compiles

The compiling program (`k0` of the same batch, the reference program) reads the PREVIOUS centre for an empty cluster:

```
... else (((centres)[(i_k)])[(i_d)]) ...
```

The failing program (`k4`) reads a vector-bucket histogram at the same place, inside the `tabulate k`/`tabulate d`
that computes each cell:

```
... else (((hist (map2 (+)) (replicate d 0f32) k (map (\x -> ...) (s_a_n)) (points))[(i_k)])[(i_d)]) ...
```

so the histogram is nested one level inside a map nest with array-typed buckets. Hoisting it to the top level (as the
reference does) compiles.

## How we found it

The k-means evolution run of 2026-11-03 (`/private/tmp/claude-501/fbench/kmeans_qqs_s1`, bugs.log): 3 of the first
67 candidate programs hit this error; the runner split each affected batch and scored the rest, charging no candidate.
Every candidate the runner emits is well typed by construction (the kingdom's types carry the legality rules) and
Futhark's own type checker accepts them; this is the first compiler fault of this kind the k-means problem has met.

## What we need

A fix in the GPU histogram code generator (or a diagnostic that names the limitation), so that a legal program is
either compiled or rejected with a message; an internal error in the middle of a batch cannot be told from our own bug.
Until then our workaround is the batch split; we will add the fix to our patch series (`futhark-rust-backend/patches`)
when it exists.
