# Measurement: the A7 kernel with the population on the GPU

**Date:** 2026-10-27. The companion to
`2026-10-27_futhark_shapes_A7.md` (the result) and
`2026-10-27_futhark_shapes_A7_learning.md` (the host-side learning curve).
This one captures the run where the **population lives in the device's typed
core** and the fitness comes back from the host — the split the paper
describes.

## The environment

| | |
|---|---|
| machine | Apple M3 Max, 16 CPUs |
| GPU | Apple's OpenCL over Metal 4 (`futhark opencl`) |
| Futhark | 0.28.1 (`/opt/homebrew/bin/futhark`) |
| rustc | 1.94.0 |
| sizes | n = m = p = 8 |
| command | `cargo run --release --features gpu --example shape_fit -- --pop 32 --gens 20 --size 8 --samples 3 --measure-elites 1` (in `phylu`) |
| seed | `0x5eed1234abcd0010` |
| program size | N = 3 (322 words per row) |

The kernel is the A7 one: the product of two matrices, computed **twice** —
once directly and once the transposed way — and the two added.

At n = m = p = 8 the reference's own traffic is 704 = 3np + 4nm + 4mp, which
is the first number the run prints; it is the check that this run is the
small-size one and not the 64³ run of the learning capture.

## The run

```
# shape reference	row 0	the incumbent, in a layout of 322 words	its own traffic 704
# shape layout	program size N = 3	11 types framed	row 322 words	population 32	2 elites	54 functions 17 terminals
# shape polish	3 child rows replaced	traffic 704
# shape measured	14 device allocations (2 of a computed size)	10500 bytes literal	tiled false
# shape gen	0	correct 1/32	illegal 0	best 0.15313	agreement 0.0312	polished 3	0.2s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	1	correct 6/32	illegal 0	best 0.15313	agreement 0.1875	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	2	correct 8/32	illegal 0	best 0.15313	agreement 0.2500	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	3	correct 6/32	illegal 0	best 0.15313	agreement 0.1875	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	4	correct 8/32	illegal 0	best 0.15313	agreement 0.2500	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	5	correct 10/32	illegal 0	best 0.15313	agreement 0.3125	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	6	correct 5/32	illegal 0	best 0.15313	agreement 0.1562	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	7	correct 8/32	illegal 0	best 0.15313	agreement 0.2500	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	8	correct 5/32	illegal 0	best 0.15313	agreement 0.1562	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	9	correct 12/32	illegal 0	best 0.15313	agreement 0.3750	polished 0	0.4s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	10	correct 5/32	illegal 0	best 0.15313	agreement 0.1562	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	11	correct 10/32	illegal 0	best 0.15313	agreement 0.3125	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	12	correct 11/32	illegal 0	best 0.15313	agreement 0.3438	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	13	correct 7/32	illegal 0	best 0.15313	agreement 0.2188	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	14	correct 8/32	illegal 0	best 0.15313	agreement 0.2500	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	15	correct 6/32	illegal 0	best 0.15313	agreement 0.1875	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	16	correct 12/32	illegal 0	best 0.15313	agreement 0.3750	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	17	correct 10/32	illegal 0	best 0.15313	agreement 0.3125	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	18	correct 9/32	illegal 0	best 0.15313	agreement 0.2812	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	19	correct 8/32	illegal 0	best 0.15313	agreement 0.2500	polished 0	0.5s
# shape measured	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
# shape gen	20	correct 8/32	illegal 0	best 0.15313	agreement 0.2500	polished 0	0.5s
# shape done	21 generations	best 0.15313	correct (summed over the generations) 163	the reference's own traffic 704
# shape best	correct true	agreement 1.0000	traffic 320 alloc 64 depth 8
# shape best	written to /tmp/shape/poster/best.fut
# shape measured best	9 device allocations (2 of a computed size)	9728 bytes literal	tiled false
```

## What the columns mean

| column | meaning |
|---|---|
| `correct` | how many of the 32 individuals compute the reference's value, bit for bit, over the input suite |
| `illegal` | how many individuals do not express at all — the layout has no room for them |
| `best` | the best fitness in the generation; 0 is perfect, so the number is the *distance* from perfect |
| `agreement` | the mean agreement over the population, the share of the value that matches the reference bit for bit |
| `polished` | how many child rows the e-graph's polish wrote into the generation |
| `measured` | the device allocations of the program the measurement was taken on |

## What the run shows

- **The e-graph's rule fires at generation 0 and never again.** The polish
  wrote 3 child rows at generation 0 and none after it. The measured cost
  falls from the reference's 14 device allocations to 9 in the same
  generation, and stays at 9 for the remaining twenty.
- **The population never becomes wholly correct.** The correct count runs
  1, 6, 8, 6, 8, 10, 5, 8, 5, 12, 5, 10, 11, 7, 8, 6, 12, 10, 9, 8, 8 — it
  oscillates between 5 and 12 of 32, and the mean agreement never passes
  0.375. The best individual, though, is correct: `agreement 1.0000`, traffic
  320, alloc 64, depth 8.
- **The measured winner is the un-tiled form** (9 device allocations, 9728
  bytes literal, `tiled false`), the same shape of win as the 64³ run.

## The reproducible command

```bash
cd ../phylu
cargo run --release --features gpu --example shape_fit -- \
    --pop 32 --gens 20 --size 8 --samples 3 --measure-elites 1
```

The run writes its best program to `/tmp/shape/poster/best.fut`: the
shape-polymorphic kernel over the interface's roles and the `entry main` at
the concrete sizes.
