# nluca generated_step through the user flow (nluca_f1, 2026-10-14 17:04–17:12Z)

**Result: at 1,048,576 rows the evolved kernel is 13 % faster than nluca's shipped kernel (21.74 against 24.97 ms per 64-step run, paired ratio 0.871 [0.870, 0.872]); at 65,536 rows it is 14 % slower (1.420 against 1.247 ms, 1.138 [1.130, 1.144]).**

- Flow: nluca's shipped generated_step plus 24 Sonnet-written kernels, converted at fuller 0ade835 (the parser fix that makes the emitted shipped kernel run at the original's speed: 1.265 / 25.05 / 25.08 against 1.255 / 25.03 / 25.05 ms), phylu 8bef457f; all operators (rule-all, rewrites 0.5, point mutation, typed crossover, Sonnet rewrite 0.1, Sonnet "beat both" 0.05); 26 individuals, 20 generations, 7 min 39 s.
- The best (gen 13 r7, fitness 0.5607) passed nluca's oracle (16 cases, 64 steps, trajectory tolerance). Text: problems/nluca_step/tuned_f1.wgsl.
- Paired timing, 10 ABBA rounds a shape, the shipped kernel as the original (logs/HGP/001/nluca_f1/paired_best_vs_shipped.tsv):

| shape | shipped, ms | tuned, ms | ratio [95 %] |
|---|---|---|---|
| 65,536 rows | 1.247 | 1.420 | 1.138 [1.130, 1.144] |
| 1,048,576 rows, case 1 | 24.971 | 21.737 | 0.871 [0.870, 0.872] |
| 1,048,576 rows, case 2 | 24.956 | 21.751 | 0.871 [0.871, 0.872] |

- The run's own single timings are unreliable at 65,536 rows (the shipped kernel timed 4.56 ms in-run against 1.27 paired), so its fitness mixed noise into the small shape; selection improved the best once (gen 13).
- Not yet done: a size-aware choice (the tuned kernel for large widths, the shipped for small), repeat seeds, and the held-out widths.
