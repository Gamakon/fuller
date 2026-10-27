# Measurement: the A7 kernel, on the GPU

**Date:** 2026-10-27. This is the captured run behind the numbers in
`docs/PLAN_futhark_shapes_product.md` §6, so that the claim has evidence in
the repository rather than in a chat.

## The environment

| | |
|---|---|
| machine | Apple M3 Max, 16 CPUs |
| GPU | Apple's OpenCL over Metal 4 (`futhark opencl`) |
| Futhark | 0.28.1 (`/opt/homebrew/bin/futhark`) |
| rustc | 1.94.0 |
| sizes | n = m = p = 64 |
| command | `cargo run --release --features wgsl --example futhark_shapes_evolve -- --generations 3 --population 8` |

The kernel is the A7 one: the product of two matrices, computed **twice** —
once directly and once the transposed way — and the two added.

## The run

```
== the reference ==
cheap cost    traffic 45056   alloc 16384   depth 64
GPU cost      15 device allocations (2 of a computed size), 188416 bytes literal, 15 shared, 10 kernels   tiled: true

== the search ==
 gen  correct     traffic     alloc  polished    improved   measured    tiled     no-gpu
   0        1       20480      4096          4           1          2        5          2
   1        7       20480      4096         10           0          3        1          4
   2        8       20480      4096          4           0          6        5          2

== the best ==
cheap cost    traffic 20480   alloc 4096   depth 64
-- the best's source --
def kernel [m][n][p] (m_n_n: [n][n]f32) (m_n_m: [n][m]f32) (m_n_p: [n][p]f32) (m_m_n: [m][n]f32) (m_m_m: [m][m]f32) (m_m_p: [m][p]f32) (m_p_n: [p][n]f32) (m_p_m: [p][m]f32) (m_p_p: [p][p]f32) : [n][p]f32 =
  map2 (\xs ys -> map2 (+) xs ys) (matmul (m_n_m) (m_m_p)) (matmul (m_n_m) (m_m_p))
GPU cost      9 device allocations (2 of a computed size), 106496 bytes literal, 7 shared, 5 kernels   tiled: false
the win       15 device allocations -> 9 (-40.0%)

== the winner, on the GPU ==
the GPU ran the program: 4096 elements, 758 agree exactly
the largest difference from the interpreter: 0.00036621094 (the GPU contracts the multiply-add, the interpreter does not)

== determinism ==
re-running with the same seed gives the same best: true
```

## What the numbers say, and what they do not

**The winner is the product computed once.** The e-graph's
transposed-product rule — `(Y' · X')' = X · Y` — rewrote the second operand
into the first, and the compiler's own CSE then removed the duplicate. The
static cost follows: the traffic falls from `3np + 4nm + 4mp` to
`np + 2nm + 2mp`, and the allocations from `np + np + nm + mp` to `np` — a
quarter, not a half.

**The GPU agrees about the allocations:** 15 device allocation sites become
9, and the literal bytes 188 416 become 106 496.

**The reference does not run on the GPU at all.** Its OpenCL kernel is too
big for the device to compile — `createKernel: newComputePipelineState
failed`, error −48 — while the winner runs ✓. The "no-gpu" column counts the
candidates the device refused. So the hard gate is not the cost: it is
whether the program *runs*, and the search's win is turning a kernel the GPU
cannot compile into one it can.

**The tiling and the allocations pull in opposite directions.** The
reference's kernel is block+register tiled; the winner's is not. So the
measurements are reported side by side rather than collapsed into one
number, and neither is allowed to stand in for the other.

**The GPU's arithmetic is not the interpreter's.** 758 of the 4096 elements
agree bit for bit, and the rest differ by at most 3.7e-4 — the OpenCL
backend contracts the multiply-add into an FMA and the interpreter does not.
The *interpreter* is the oracle for the sequential C backend, where it is
exact; on the GPU the oracle is the backend's own reference run.

**The byte total is a lower bound.** The ImpGPU holds several alternative
versions of the same work, chosen at run time by thresholds, so summing the
allocation sites overstates what one run allocates; and two of the sites have
a size that is computed at run time, so their bytes are not in the total at
all. The count of the allocation sites is the exact figure, and it is the
first key of the cost's ordering.
