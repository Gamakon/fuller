# Hall of fame: Gram kernels (H = ΦᵀΦ), run fit_engine5, 2026-10-11

Every kernel here is correct on all cases and the pair criterion, and runs the
Gram product about seven times faster than the shipped kernel. Re-timed at the
fitness's three shapes, seven rounds each (`logs/HGP/001/retime_shapes_*.log`):

| kernel | HFF in the run | ms median at d 2048 / 4096 / 8192 | speed-up over shipped |
|---|---|---|---|
| shipped | — | 1009 / 1005 / 1013 | 1 |
| champion_g25.wgsl | 0.3204 | 155.1 / 146.3 / 152.3 | 6.51 / 6.87 / 6.65 |
| champion_g15.wgsl | 0.3228 | 150.0 / 155.4 / 152.1 | 6.73 / 6.47 / 6.66 |
| draft_best_g0.wgsl (model-written, generation 0) | 0.3300 | 150.0 / 146.3 / 152.9 | 6.73 / 6.87 / 6.62 |
| offspring_g2_r15.wgsl | 0.3287 | 149.4 / 145.4 / 152.8 | 6.75 / 6.92 / 6.63 |

The four are the same speed within 4 %. The speed is the model-written draft's;
the evolution kept it and did not improve on it measurably.
