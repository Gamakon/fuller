TASK. Write a compute kernel that advances a reservoir (an echo-state network of `width` neurons, each with `degree` = 10 incoming edges) by one time step, exactly as the reference kernel below does, but faster. Every edge weight is generated on the device from a hash as it is used: no weight is stored anywhere. Written from Noctiluca's contract (noctiluca-dev/results/nluca/cases/README.md) and the kernel's source (noctiluca-dev rust/nluca/src/gpu.rs, `generated_step`).

CONTRACT (fixed, the harness binds exactly these nine buffers; every one is declared `read_write` as CubeCL emits them, but only `out` may be written):
```wgsl
@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>; // sources: width*degree, slot s of row r at r*degree + s (the source neuron of edge s of row r)
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>; // state: width, the state x before the step
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>; // codebook: 1024 levels, indexed by hash >> 22
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>; // drive: width, the input, the same at every step
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>; // out: width, the state after the step (written)
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>; // ints: width, key_lo, key_hi, mask
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>; // offset: band offset (0), noise_lo, noise_hi, step t
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>; // reals: loc, scale, leak, u, sigma
@group(0) @binding(8) var<storage, read_write> info: array<u32>;            // CubeCL's metadata, 16 zero words, unused
```

SEMANTICS. With `fmix32(h)` MurmurHash3's 32-bit finaliser (`h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16`, wrapping u32 arithmetic throughout), for every row `r = id + offset[0]` with `r < ints[0]` (id the thread's global index):
- `acc_z = sum over slot s in [0, 10) of codebook[fmix32(fmix32(ints[1] ^ r) ^ (ints[2] ^ s)) >> 22] * state[sources[r*10 + s]]` (the reference sums it Kahan-compensated: this sum feeds back into itself every step, so its error compounds over the steps);
- `acc_x = sum over s of state[sources[r*10 + s]]`;
- `z = codebook[fmix32(fmix32(offset[1] ^ r) ^ (offset[2] ^ offset[3])) >> 22]` (the noise, a pure function of row and step);
- `pre = reals[1] * acc_z + reals[0] * acc_x + drive[r] + reals[4] * z`;
- `out[r] = (1 - reals[2]) * state[r] + reals[2] * tanh(pre)`, with tanh(x) = 1 for x > 43 (Metal's tanh misbehaves above 43).
Rows at or past `ints[0]` must not be read or written. A row is written by exactly one thread, and nothing but `out` is written. The kernel must not read `out` (step t + 1 reads step t's `out` as its `state`, in the other buffer of a ping-pong pair).

DISPATCH. The harness dispatches ceil(width / X) x 1 x 1 workgroups of the kernel's `@workgroup_size(X, 1, 1)` (the reference: 256), 64 steps a case, with offset[3] = t before step t; width runs from 4,096 to 1,048,576.

ENTRY POINT. `@compute @workgroup_size(X, 1, 1) fn generated_step(...)` with any builtins you need; helper functions are allowed.

HOW IT IS SCORED. The state after steps 1, 8 and 64 is compared with an f64 reference's: at every checkpoint the largest |got - expected| over the RMS of the expected state must be below the case's tolerance (ten times the shipping f32 kernel's own worst error, at least 1e-4); one failing checkpoint of one case and the kernel is infeasible. A state held in f16 fails every case. A feasible kernel is timed on three widths (65,536 and two of 1,048,576) against the reference kernel; faster is better.

CONSTRAINTS. Declare exactly the nine bindings above, with these names, types and binding numbers, the unused `info` among them: a kernel that renames, retypes or drops one is not read into the population. No other global variables except var<workgroup> arrays (at most 16 KiB in total). workgroupBarrier() only in uniform control flow. Every loop must terminate. f32 arithmetic for the state; the hash in wrapping u32.

## Strategies
- the reference kernel's organisation of the work (one thread a row), with any improvement you see
- the ten edges unrolled, the hash of each slot computed once and the key words read once a thread
- more rows per thread, each thread's rows adjacent so the reads of drive, state and out are contiguous
- the codebook (4 KiB) staged in var<workgroup> memory once a workgroup, between workgroupBarrier calls
- the same work with fewer instructions: repeated loads of ints, offset and reals hoisted, recomputed values removed
- a structure unlike the reference kernel: choose any correct organisation of the work you believe is fastest on an Apple M3 Max GPU under Metal
