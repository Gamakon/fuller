TASK. Write a compute kernel that assigns each of a band of points to its most similar centroid, exactly as the reference kernel below does, but faster: for every point, the dot product of its key with each of C centroid keys over D_proj dimensions, and the index of the largest. Written from UltraDim's k-means (qdrant lib/projection-voting-kmeans/src/lib.rs: the kernel `build_cosine_assign_pipeline`, lines 4732-4769; its driver `assign_cosine_resident_banded`, lines 1996-2062). Keys are unit vectors, so the dot product is the cosine similarity.

CONTRACT (fixed, the harness binds exactly these four buffers):
```wgsl
@group(0) @binding(0) var<storage, read> point_keys: array<f32>;            // n_band x D_proj, row-major: the band's points, point l at [l*D_proj, l*D_proj + D_proj)
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;         // C x D_proj, row-major: centroid ci at [ci*D_proj, ci*D_proj + D_proj)
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;     // N: the whole output, written at the global row
@group(0) @binding(3) var<uniform> params: vec4<u32>;                        // x = n_band, y = C, z = D_proj, w = row_off
```

SEMANTICS. For every local point l < params.x (l the thread's global index in the reference), with g = params.w + l: assignments[g] = the ci in [0, C) maximising sum over p in [0, D_proj) of point_keys[l*D_proj + p] * centroid_keys[ci*D_proj + p]; among equal sums the smallest ci (the reference keeps the first: `if (sim > best_sim)`, starting from best_sim = -3.4e38, best = 0). Points at or past params.x must not be written. A point is written by exactly one thread, and nothing but `assignments` is written. The kernel must not read `assignments`.

DISPATCH. The harness splits the N points into contiguous bands of at most its binding limit (qdrant's create_raw_bands) and dispatches ceil(n_band / X) x 1 x 1 workgroups of the kernel's `@workgroup_size(X, 1, 1)` (the reference: 256) once per band, params = (n_band, C, D_proj, row_off). D_proj is up to 2,048 (2,048 in production), C up to 65,536 (production: 20 to 9,000), N up to hundreds of thousands.

ENTRY POINT. `@compute @workgroup_size(X, 1, 1) fn main(...)` with any builtins you need; helper functions are allowed.

HOW IT IS SCORED. The expected assignments are the reference kernel's own, on generated unit-length keys (no hand-made reference). An index that differs from the reference's is accepted only when its dot product is within 2 * D_proj * f32 epsilon of the reference index's (a near tie, decided by rounding); any other difference makes the kernel infeasible. Reordering the sum is fine. A feasible kernel is timed against the reference at N 50,000 and 262,144 with C 20, and N 18,000 with C 9,000, all at D_proj 2,048; faster is better.

CONSTRAINTS. Declare exactly the four bindings above, with these names, types and binding numbers: a kernel that renames, retypes or drops one is not read into the population. No other global variables except var<workgroup> arrays (at most 16 KiB in total). workgroupBarrier() only in uniform control flow (never after a conditional return; guard with if-blocks instead). Every loop must terminate: bound loops by params.y, params.z or a literal. f32 arithmetic.

## Strategies
- the reference kernel's organisation of the work (one thread a point), with any improvement you see
- the point's key read as vec4<f32> and the dot product taken four lanes at a time
- a tile of centroid keys staged in var<workgroup> memory once a workgroup and shared by its points, between workgroupBarrier calls
- several points per thread, so each centroid value read is used for all of them
- several centroids per pass over the point's key, their partial sums in registers
- a structure unlike the reference kernel: choose any correct organisation of the work you believe is fastest on an Apple M3 Max GPU under Metal
