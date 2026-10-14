// UltraDim k-means cosine assignment, the shipped kernel: the WGSL string of
// build_cosine_assign_pipeline in qdrant lib/projection-voting-kmeans/src/lib.rs,
// lines 4743-4766 (fn main at line 4748), copied verbatim; the reference
// kernel of problems/kmeans_assign (the oracle of its cases and its timing
// denominator). Dispatched by assign_cosine_resident_banded (lib.rs:2042-2061):
// one dispatch per N-band, ceil(n_band / 256) x 1 x 1 workgroups.
            @group(0) @binding(0) var<storage, read> point_keys: array<f32>;
            @group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
            @group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
            @group(0) @binding(3) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let local = gid.x;
                let n_band = params.x;
                let c = params.y;
                let dp = params.z;
                let row_off = params.w;
                if (local >= n_band) { return; }
                let global = row_off + local;
                var best: u32 = 0u;
                var best_sim: f32 = -3.4e38;
                for (var ci: u32 = 0u; ci < c; ci = ci + 1u) {
                    var sim: f32 = 0.0;
                    for (var p: u32 = 0u; p < dp; p = p + 1u) {
                        sim = sim + point_keys[local * dp + p] * centroid_keys[ci * dp + p];
                    }
                    if (sim > best_sim) { best_sim = sim; best = ci; }
                }
                assignments[global] = best;
            }
