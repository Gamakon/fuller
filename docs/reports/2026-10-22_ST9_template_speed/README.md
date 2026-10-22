# ST9: the regenerated volumepassage as fast as the shipped shader (2026-10-22)

**The encoded volumepassage now takes 0.983x the shipped shader's GPU time** (64 x 64, t 5, one submit, 5 rounds of 41
interleaved pairs, one command buffer a pair; median of 205 ratios; 13:56Z 0.9829, 18:42Z 0.9831). Before: 1.042-1.052.
Shipped/shipped and encoded/encoded read 1.000. 0 differing channels against the shipped shader.

| cause | cost (probe against the encoded text) | fix |
|---|---|---|
| the tile offset of a split dispatch | 2.9 % | `override DISPATCH_TILED: bool = true;` in the image prologue; the entry's lets are `select(<id>_tile, <id>_tile + offset, DISPATCH_TILED)`; `ImageKernel` creates the pipeline with 0 for a one-submit plan, 1 for several, and remakes it when a plan of the other kind is imposed. After it, removing the tile binding outright: 1.004 (`time_loop_spellings.log`, notile) |
| the counted loop's spelling `k = 4294967295u; loop { k = k + 1u; if (k >= n) { break; } ... }` | 4.3 % (shadow / encoded 0.957) | `kc = 0u; loop { k = kc; if (k >= n) { break; } kc = k + 1u; ... }`: no continuing block (no `loop_init` in naga's MSL), same values of `k` inside, after, and after a break, a `continue` takes the step |
| remaining `lit_*` calls | 0.0 % (bare / encoded 1.000) | none needed |
| `zero_vec3_f32()`, `col_axis(gid)`, the bounds as literals | under 0.6 % each, mixed signs | none needed |

The loop spelling also beats naga's `for` form (shipped's) here: shadow 0.978 against the shipped, `for` 0.999.

Hang reproducer (`hang_reproducer_proof.log`, `proof.sh`): the shadow spelling of gene 583 runs in 3.5 ms through the
Metal API and passes through wgpu (VERDICT ok); the loop_init controls are still killed by the GPU's hang detector
(kIOGPUCommandBufferCallbackErrorHang, about 0.77 s); the GPU answered the probe after them.

MSL: `shipped.metal`, `encoded_one_submit.metal` (as wgpu 30 writes it on this machine, `fuller::wgsl::target::metal_source`,
`examples/st9_msl.rs`), `encoded_old_loop.metal`. Probes: `probes.py` (text edits for measurement only, never kernels of the kingdom).
