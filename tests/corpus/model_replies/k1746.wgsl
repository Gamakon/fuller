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
    let valid = local < n_band;
    let base = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 16u) {
        var acc: array<f32, 16>;
        for (var k = 0u; k < 16u; k += 1u) { acc[k] = 0.0; }
        for (var dt: u32 = 0u; dt < dp; dt += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 2048u; i += 64u) {
                let idx = i + lid;
                let k = idx / 128u;
                let d = idx % 128u;
                let ci = cb + k;
                let dd = dt + d;
                var v: f32 = 0.0;
                if (ci < c && dd < dp) { v = centroid_keys[ci * dp + dd]; }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var d = 0u; d < 128u; d += 4u) {
                let dd = dt + d;
                var pv = vec4<f32>(0.0);
                if (valid) {
                    if (dd + 3u < dp) {
                        pv = vec4<f32>(point_keys[base + dd], point_keys[base + dd + 1u], point_keys[base + dd + 2u], point_keys[base + dd + 3u]);
                    } else {
                        if (dd < dp) { pv.x = point_keys[base + dd]; }
                        if (dd + 1u < dp) { pv.y = point_keys[base + dd + 1u]; }
                        if (dd + 2u < dp) { pv.z = point_keys[base + dd + 2u]; }
                        if (dd + 3u < dp) { pv.w = point_keys[base + dd + 3u]; }
                    }
                }
                for (var k = 0u; k < 16u; k += 1u) {
                    let o = k * 128u + d;
                    acc[k] += dot(pv, vec4<f32>(tile[o], tile[o + 1u], tile[o + 2u], tile[o + 3u]));
                }
            }
        }
        for (var k = 0u; k < 16u; k += 1u) {
            let ci = cb + k;
            if (ci < c && acc[k] > best_sim) { best_sim = acc[k]; best = ci; }
        }
    }
    if (valid) { assignments[row_off + local] = best; }
}
