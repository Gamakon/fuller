@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let active = local < n_band;
    let base = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < dp; d0 += 256u) {
            workgroupBarrier();
            for (var i = 0u; i < 32u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx >> 8u;
                let k = idx & 255u;
                let ci = c0 + cc;
                let d = d0 + k;
                var v: f32 = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            let kmax = min(256u, dp - d0);
            if (active) {
                for (var k = 0u; k < kmax; k += 1u) {
                    let p = point_keys[base + d0 + k];
                    a0.x = a0.x + p * tile[k];
                    a0.y = a0.y + p * tile[256u + k];
                    a0.z = a0.z + p * tile[512u + k];
                    a0.w = a0.w + p * tile[768u + k];
                    a1.x = a1.x + p * tile[1024u + k];
                    a1.y = a1.y + p * tile[1280u + k];
                    a1.z = a1.z + p * tile[1536u + k];
                    a1.w = a1.w + p * tile[1792u + k];
                }
            }
        }
        if (c0 + 0u < c && a0.x > best_sim) { best_sim = a0.x; best = c0 + 0u; }
        if (c0 + 1u < c && a0.y > best_sim) { best_sim = a0.y; best = c0 + 1u; }
        if (c0 + 2u < c && a0.z > best_sim) { best_sim = a0.z; best = c0 + 2u; }
        if (c0 + 3u < c && a0.w > best_sim) { best_sim = a0.w; best = c0 + 3u; }
        if (c0 + 4u < c && a1.x > best_sim) { best_sim = a1.x; best = c0 + 4u; }
        if (c0 + 5u < c && a1.y > best_sim) { best_sim = a1.y; best = c0 + 5u; }
        if (c0 + 6u < c && a1.z > best_sim) { best_sim = a1.z; best = c0 + 6u; }
        if (c0 + 7u < c && a1.w > best_sim) { best_sim = a1.w; best = c0 + 7u; }
    }
    if (active) {
        assignments[row_off + local] = best;
    }
}
