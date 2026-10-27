# The image study through Futhark, parked (2026-10-27)

**Parked at Andrew's request; nothing is running.** Futhark can express both cuneus shaders exactly, and their hand-written references draw the WGSL originals' pictures except for fast-math differences. Evolution from random did not draw any target picture in the runs we made: volumepassage stayed at a black picture, cuneus was stopped at generation 8, and bugberman at generation 5. The staged fitness (quality only until the picture is perfect, then the two-level HFF) is in the shared runner code, for the pathfinder benchmark.

## What was built

| piece | where | commit |
|---|---|---|
| The pixel family: a picture as a map over pixels of a scalar program. Scope, `let`, bounded `loop`, `block` and result arity are types, so every gene is legal; witness tests P1-P9; one instance a WGSL shader (its uniforms and literals read from the source) | `kingdoms/futhark_shapes/gen_tables.py`, `pixel/{volumepassage,cuneus,image}/` | 8834c828, d17ed1e9, fa751a1b |
| Loader, program text, work bound and tiles under the per-submit cap, half-float store rounding, e-graph (IEEE identities, bit-identical under `futhark c`), saturated-fold bridge, hand-written references (volumepassage 454 nodes, cuneus 2,119 nodes), host interpreter (agrees with `futhark c` to 1e-4), random genes with the half head draw | `src/futhark_pixel.rs` | 8834c828, d17ed1e9, a95d6f3c, fa751a1b |
| The image-target instance: no shader and no uniforms; position in, colour out; any PPM picture is a target. bugberman at 24x22, 96x88, 128x117 and 512x469 | `pixel/image/`, `tests/corpus/targets/` | fa751a1b |
| sRGB display space in PNGs and in `image::compare` | `src/wgsl/image.rs` | 17800777 |
| OCR (tesseract), PPM target reader, largest-colour share, `quality_inputs` | `src/wgsl/image.rs` | d17ed1e9, fa751a1b |
| The runner: `fuller::telemetry` stream, kill trigger, hall of fame with pictures and OCR, checkpoint and `--resume`, ledger with lineage, two regimes (petri, champion) | `examples/futhark_pixel_evolve.rs` | d17ed1e9 and later |
| Staged fitness, shared: `runner::FitnessStage`, `staged_hff`, `EventKind::StageChange` | `src/runner.rs`, `src/telemetry.rs` | 922ec1a6 |

## What was measured

| measurement | conditions | result |
|---|---|---|
| volumepassage reference against the WGSL original | Metal through the shim, 128x128, t = 5, linear values | 13 of 16,384 pixels over 1/255, mean abs difference 2.0e-5 |
| cuneus reference against the WGSL original | 96x48, t = 5, display space | 214 of 4,608 pixels over 1/255, mean abs difference 1.4e-3 (the SDF normal's 0.001 finite difference amplifies fast math) |
| OCR control, cuneus original and reference | 96x48 to 512x256, raw and thresholded | neither reads exactly CUNEUS; best "CUNELS" or "CUNELY", edit distance 1 to 2 |
| Timing noise, the reference against itself | median of 9 runs a process / fastest of 40 runs a process | ratio 0.62 to 3.6 / 0.993 to 1.006 on a quiet machine (0.68 to 2.19 with other agents building) |
| volumepassage petri dish | 30 min, population 24, 27 generations, 610 candidates, 0 bugs, linear-space comparison | best stayed a black picture (quality 0.847, speed ratio 0.05); the reference scores HFF 0.500 against the best's 0.547 |
| volumepassage champion | 2-minute trials | e-graph variants measured 0.997 to 1.002 of the reference's time; no faster equivalent |
| Random tree size, cuneus layout, N = 60, 1,000 genes | uniform head draw / half draw | median 3,977 / 4 nodes; host compile and link of a half-draw candidate 1.10 to 1.22 s |
| Colour gate, image target, 1,000 genes, half draw, 24x22 | host interpreter | 99.5 % not uniform black; 97.5 % use more than one of 64 colour bins (median 16: mostly the position gradient); bugberman's mean colour scores quality angle 0.790 against black's 0.909 |
| Futhark branch compiler | random cuneus candidates | about 1 in 8 hit an internal error ("Ratio has zero denominator"); logged in `bugs.log` and scored worst |

## The fixes found on the way

1. **Our pictures were darker than the screen shows (sRGB).** cuneus draws into an sRGB surface, so the hardware encodes the stored value; we saved and compared the linear value (0.03 shown as 8/255, not 48/255). PNGs and `compare` now use display space. Numbers before 17800777 are linear, including `2026-10-22_ST9_*` and every phylu image-fitness HFF.
2. **Random programs drew black.** The block terminal was "black", so half of all random genes drew it at the root. The image instance's block terminal is the position's colour, `nx` and `ny` are terminals, and head cells are drawn half functions, half terminals (the uniform draw filled the layout).
3. **Black programs won on speed alone.** A picture that does not depend on the pixel compiles to a fill and ran 4x faster than the gradient yardstick. A floor at the yardstick (0d0de23a) was replaced by the staged fitness (922ec1a6): stage 1 runs no timing.
4. **Clamped relative errors gave no gradient.** Every random picture scored 1, so they now map e to e/(1+e).
5. **The branch compiler has no f32-to-i64 conversion in a kernel.** Tile offsets stay f32.

## How to resume

- Runs and logs: `/private/tmp/claude-501/fpix/` (`petri/`, `cuneus_petri/`, `bugberman_petri_*`, `cuneus_reference/`, `reference/`).
- An image target from random, stage 1 first:

      cargo run --release --features gpu,wgsl --example futhark_pixel_evolve -- --shader image --target bugberman \
          --regime petri --minutes 90 --pop 24 --checkpoint-every 5 --seed 1 --out OUT

  Watch it with `phylu-sr-watch --follow OUT --kingdom wgsl`; stop it by deleting `OUT/killme.trigger`; continue it with `--resume`. Take the GPU lock (`/private/tmp/claude-501/locks/gpu`) first.
- A new picture target: write `tests/corpus/targets/<name>_<w>x<h>.ppm` at the training, held-out and report sizes (PIL LANCZOS on the sRGB values). The loader logs the largest colour's share.
- A shader target: `--shader volumepassage|cuneus` (petri or champion); the reference comparison is `futhark_pixel_reference --shader cuneus --sizes 96x48,512x256`.
- Open points: stage 1 has not yet reached stage 2 on any target; the champion regime has had only 2-minute trials; the branch compiler's internal error is not reported upstream.
