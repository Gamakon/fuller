@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = local < n_band;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0: u32 = 0u; c0 < c; c0 += 16u) {
        var acc: array<f32, 16>;
        for (var r = 0u; r < 16u; r += 1u) { acc[r] = 0.0; }
        for (var d0: u32 = 0u; d0 < dp; d0 += 64u) {
            for (var i = 0u; i < 256u; i += 128u) {
                let idx = i + lid;
                let r = idx / 16u;
                let k = idx % 16u;
                let ci = c0 + r;
                let base = d0 + k * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let o = ci * dp + base;
                    if (base + 0u < dp) { v.x = centroid_keys[o]; }
                    if (base + 1u < dp) { v.y = centroid_keys[o + 1u]; }
                    if (base + 2u < dp) { v.z = centroid_keys[o + 2u]; }
                    if (base + 3u < dp) { v.w = centroid_keys[o + 3u]; }
                }
                tile[idx] = v;
            }
            var pt: array<vec4<f32>, 16>;
            for (var k = 0u; k < 16u; k += 1u) {
                let base = d0 + k * 4u;
                var v = vec4<f32>(0.0);
                if (valid) {
                    let o = local * dp + base;
                    if (base + 0u < dp) { v.x = point_keys[o]; }
                    if (base + 1u < dp) { v.y = point_keys[o + 1u]; }
                    if (base + 2u < dp) { v.z = point_keys[o + 2u]; }
                    if (base + 3u < dp) { v.w = point_keys[o + 3u]; }
                }
                pt[k] = v;
            }
            workgroupBarrier();
            for (var r = 0u; r < 16u; r += 1u) {
                var s0: f32 = 0.0;
                var s1: f32 = 0.0;
                for (var k = 0u; k < 16u; k += 2u) {
                    s0 += dot(pt[k], tile[r * 16u + k]);
                    s1 += dot(pt[k + 1u], tile[r * 16u + k + 1u]);
                }
                acc[r] += s0 + s1;
            }
            workgroupBarrier();
        }
        for (var r = 0u; r < 16u; r += 1u) {
            let ci = c0 + r;
            if (ci < c) {
                let s = acc[r];
                if (s > best_sim) { best_sim = s; best = ci; }
            }
        }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
