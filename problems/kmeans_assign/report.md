# kmeans_assign: tuned on this machine (04:48:19Z to 05:09:26Z)

**No kernel was shown faster at every timing shape** (largest upper bound 1.090, timed alone); `tuned.wgsl` is the shipped kernel.

## The run's best against the shipped kernel (paired, 10 rounds, timed alone 05:13Z-05:18Z)

| timing shape | shipped, ms | tuned, ms | ratio [95 %] |
|---|---|---|---|
| n50000_c20_d2048 | 27.249 | 29.450 | 1.083 [1.070, 1.084] |
| n262144_c20_d2048 | 117.833 | 127.673 | 1.087 [1.081, 1.090] |
| n18000_c9000_d2048 | 5461.485 | 3712.438 | 0.679 [0.654, 0.690] |

The flow's own timing of this step (05:04Z-05:09Z) overlapped another GPU job for about 20 s (05:05:18-05:05:37Z) and is discarded: it gave 1.103, 1.091 and 0.510, the shipped kernel at 8,206 ms on the largest case against 5,461 ms alone. The choice is unchanged: the best is slower at two shapes, so `tuned.wgsl` is the shipped kernel.

## Step 2: the emitted shipped kernel against the shipped text (paired, 10 rounds)

| timing shape | shipped, ms | emitted, ms | ratio [95 %] |
|---|---|---|---|
| n50000_c20_d2048 | 27.802 | 27.882 | 1.003 [1.000, 1.006] |
| n262144_c20_d2048 | 119.834 | 120.477 | 1.002 [0.993, 1.007] |
| n18000_c9000_d2048 | 5660.975 | 5649.805 | 0.997 [0.993, 1.004] |

## Run

- `wgsl_fit`, seed 1, 4 generations, section crossover; logs in `/Users/andrewmorgan/Dev/gamakon/fuller/logs/HGP/001/tune_kmeans`.
