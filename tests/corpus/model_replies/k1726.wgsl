@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(128, 1, 1)
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
        var s: array<f32, 8>;
        for (var k = 0u; k < 8u; k += 1u) { s[k] = 0.0; }
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 8u; i += 1u) {
                let ci = c0 + i;
                let d = d0 + lid;
                var v = 0.0;
                if (ci < c && d < dp) { v = centroid_keys[ci * dp + d]; }
                tile[i * 128u + lid] = v;
            }
            workgroupBarrier();
            if (active) {
                let rem = min(128u, dp - d0);
                for (var d = 0u; d < 128u; d += 1u) {
                    if (d < rem) {
                        let pv = point_keys[base + d0 + d];
                        for (var k = 0u; k < 8u; k += 1u) {
                            s[k] = s[k] + pv * tile[k * 128u + d];
                        }
                    }
                }
            }
        }
        for (var k = 0u; k < 8u; k += 1u) {
            let ci = c0 + k;
            if (ci < c && s[k] > best_sim) { best_sim = s[k]; best = ci; }
        }
    }
    if (active) { assignments[row_off + local] = best; }
}
