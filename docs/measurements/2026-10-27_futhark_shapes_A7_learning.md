# Measurement: the A7 kernel's learning curve

**Date:** 2026-10-27. The companion to
`2026-10-27_futhark_shapes_A7.md`, which captures the *result* (three
generations, eight individuals). This one captures the *learning* (twenty
generations, thirty-two individuals), so the poster's claims about the rate of
learning have a run behind them.

## The environment

| | |
|---|---|
| machine | Apple M3 Max, 16 CPUs |
| GPU | Apple's OpenCL over Metal 4 (`futhark opencl`) |
| Futhark | 0.28.1 (`/opt/homebrew/bin/futhark`) |
| rustc | 1.94.0 |
| sizes | n = m = p = 64 |
| command | `cargo run --release --features wgsl --example futhark_shapes_evolve -- --generations 20 --population 32` |
| seed | `0x5eed1234abcd0010` |

The kernel is the A7 one: the product of two matrices, computed **twice** —
once directly and once the transposed way — and the two added.

## The run

```
== the reference ==
cheap cost    traffic 45056   alloc 16384   depth 64
GPU cost      15 device allocations (2 of a computed size), 188416 bytes literal, 15 shared, 10 kernels   tiled: true

== the search ==
 gen  correct     traffic     alloc   polished    improved   measured    tiled     no-gpu
   0        1       20480      4096          4           1          2        1          2
   1       12       20480      4096          4           0          4        1          4
   2       24       20480      4096          4           0          8        0          0
   3       31       20480      4096          4           0          8        0          0
   4       32       20480      4096          4           0          8        0          0
   5       32       20480      4096          4           0          8        0          0
   6       30       20480      4096          4           0          8        0          0
   7       31       20480      4096          4           0          8        0          0
   8       28       20480      4096          4           0          8        0          0
   9       29       20480      4096          4           0          8        0          0
  10       28       20480      4096          4           0          8        0          0
  11       31       20480      4096          4           0          8        0          0
  12       31       20480      4096          4           0          8        0          0
  13       31       20480      4096          4           0          8        0          0
  14       32       20480      4096          4           0          8        0          0
  15       32       20480      4096          4           0          8        0          0
  16       32       20480      4096          4           0          8        0          0
  17       30       20480      4096          4           0          8        0          0
  18       31       20480      4096          4           0          8        0          0
  19       30       20480      4096          4           0          8        0          0

== the best ==
cheap cost    traffic 20480   alloc 4096   depth 64
GPU cost      9 device allocations (2 of a computed size), 106496 bytes literal, 7 shared, 5 kernels   tiled: false
the win       15 device allocations -> 9 (-40.0%)

== the winner, on the GPU ==
the GPU ran the program: 4096 elements, 758 agree exactly
the largest difference from the interpreter: 0.00036621094 (the GPU contracts the multiply-add, the interpreter does not)
```

## What the columns are

* **correct** — how many of the 32 individuals compute the reference's value,
  bit for bit, over the whole input suite.
* **traffic / alloc** — the best individual's static cost.
* **polished** — how many candidates the e-graph offered for the elites.
* **improved** — how many of them were actually cheaper than their elite.
* **measured** — how many correct individuals were compiled for the GPU *and
  ran* on it.
* **tiled** — how many of the measured ones the compiler tiles with the
  block+register tiler. (This column was wrong in the first capture of
  `2026-10-27_futhark_shapes_A7.md`: it subtracted the tiled ones from a fixed
  denominator, so it reported the un-tiled. It is fixed here and the A7 capture
  was re-run.)
* **no-gpu** — how many the device **refused**: the kernel compiled, and the
  device would not run it.

## What the numbers say

**The correctness landscape is walkable.** The seeded population holds exactly
one correct program — the reference. By generation 4 it holds all 32.

**The unrunnable form leaves the population.** The `no-gpu` column is 2 at
generation 0, 4 at generation 1 and **0 from generation 2 on**, with every
measured candidate running from then on. And from generation 2 the compiler
stops tiling the population at all: the population has converged onto the
un-tiled, runnable form. The search's win is not merely a cheaper kernel; it
is a kernel that runs at all.

**The cost axis is not the GEP's.** The only thing that moved the cost was the
e-graph's transposed-product rule, and it did so in **generation 0**
(`improved 1` there, and 0 in every generation after). Twenty generations of
GEP never beat it. This is a design consequence, not a surprise: the e-graph's
rules are equalities by construction, so every candidate it offers is equal to
its parent and it can only move the cost axis; the GEP's operators change the
program, so they walk the correctness landscape.

**And then a plateau.** From generation 4 the correct count oscillates between
28 and 32 of 32: the search spends the rest of its budget on diversity, not on
improvement. The plateau is not broken here, and it is where the next work is.
