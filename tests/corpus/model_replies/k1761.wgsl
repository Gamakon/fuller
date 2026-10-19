@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    if (local < n_band) {
        let pb = local * dp;
        let dp4 = dp & 0xFFFFFFFCu;
        var best: u32 = 0u;
        var best_sim: f32 = -3.4e38;
        for (var ci: u32 = 0u; ci < c; ci += 4u) {
            let c0 = ci;
            let c1 = min(ci + 1u, c - 1u);
            let c2 = min(ci + 2u, c - 1u);
            let c3 = min(ci + 3u, c - 1u);
            let b0 = c0 * dp;
            let b1 = c1 * dp;
            let b2 = c2 * dp;
            let b3 = c3 * dp;
            var s0: f32 = 0.0;
            var s1: f32 = 0.0;
            var s2: f32 = 0.0;
            var s3: f32 = 0.0;
            for (var p: u32 = 0u; p < dp4; p += 4u) {
                let a0 = point_keys[pb + p];
                let a1 = point_keys[pb + p + 1u];
                let a2 = point_keys[pb + p + 2u];
                let a3 = point_keys[pb + p + 3u];
                s0 = s0 + a0 * centroid_keys[b0 + p] + a1 * centroid_keys[b0 + p + 1u] + a2 * centroid_keys[b0 + p + 2u] + a3 * centroid_keys[b0 + p + 3u];
                s1 = s1 + a0 * centroid_keys[b1 + p] + a1 * centroid_keys[b1 + p + 1u] + a2 * centroid_keys[b1 + p + 2u] + a3 * centroid_keys[b1 + p + 3u];
                s2 = s2 + a0 * centroid_keys[b2 + p] + a1 * centroid_keys[b2 + p + 1u] + a2 * centroid_keys[b2 + p + 2u] + a3 * centroid_keys[b2 + p + 3u];
                s3 = s3 + a0 * centroid_keys[b3 + p] + a1 * centroid_keys[b3 + p + 1u] + a2 * centroid_keys[b3 + p + 2u] + a3 * centroid_keys[b3 + p + 3u];
            }
            for (var p: u32 = dp4; p < dp; p += 1u) {
                let a = point_keys[pb + p];
                s0 = s0 + a * centroid_keys[b0 + p];
                s1 = s1 + a * centroid_keys[b1 + p];
                s2 = s2 + a * centroid_keys[b2 + p];
                s3 = s3 + a * centroid_keys[b3 + p];
            }
            if (s0 > best_sim) { best_sim = s0; best = c0; }
            if (ci + 1u < c && s1 > best_sim) { best_sim = s1; best = c1; }
            if (ci + 2u < c && s2 > best_sim) { best_sim = s2; best = c2; }
            if (ci + 3u < c && s3 > best_sim) { best_sim = s3; best = c3; }
        }
        assignments[row_off + local] = best;
    }
}
