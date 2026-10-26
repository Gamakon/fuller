# nluca_step cases

The case files are not copied into fuller. `constraint.toml` and `timing.toml`
point at the directory Noctiluca writes them to:
`/Users/andrewmorgan/Dev/gamakon/noctiluca-dev/results/nluca/cases` (written by
`rust/nluca/examples/cases.rs`; its `README.md` is the contract and
`manifest.jsonl` lists the cases). The manifests here were written from
`manifest.jsonl`.

Each case directory must hold these files, raw little-endian:

| file | binding | holds |
|---|---|---|
| `sources.u32.bin` | 0 `sources` | width x degree u32, slot s of row r at r*degree+s |
| `state.f32.bin` | 1 `state` | width f32, the starting state |
| `codebook.f32.bin` | 2 `codebook` | 1024 f32 levels |
| `drive.f32.bin` | 3 `drive` | width f32, the same at every step |
| (made by the harness) | 4 `out` | width f32, the ping-pong partner of `state` |
| `ints.u32.bin` | 5 `ints` | 4 u32: width, key_lo, key_hi, mask |
| `offset.u32.bin` | 6 `offset` | 4 u32: band offset 0, noise_lo, noise_hi, step 0 (the harness sets word 3 to t before step t) |
| `reals.f32.bin` | 7 `reals` | 5 f32: loc, scale, leak, u, sigma |
| (made by the harness) | 8 `info` | CubeCL's metadata array, 16 zero words (unused by generated_step) |
| `expected_step001.f32.bin`, `expected_step008.f32.bin`, `expected_step064.f32.bin` | | width f32 each: the f64 reference's state after steps 1, 8 and 64, rounded to f32 |

A case's `params` are `width`, `degree` (10), `steps` (64) and `tol` (the
manifest's `tolerance_max_rel_err`). A case passes when, at every checkpoint,
max |got - expected| / RMS(expected) <= tol.

Timing uses the two 1,048,576-wide cases and w65536_k1, because the fitness has
three timing columns and Noctiluca has written two cases at 1,048,576. A third
case at that width or larger (4M, 8M: more rows in the plan of
`rust/nluca/examples/cases.rs`) replaces w65536_k1 in `timing.toml`.
