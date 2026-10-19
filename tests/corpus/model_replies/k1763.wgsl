@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = gid.x < n_band;
    var l = gid.x;
    if (!valid) { l = 0u; }
    let pbase = l * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    let cc = lid >> 4u;
    let q = lid & 15u;

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            {
                let ci = c0 + cc;
                let p = d0 + q * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let b = ci * dp + p;
                    if (p + 3u < dp) {
                        v = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
                    } else {
                        if (p < dp) { v.x = centroid_keys[b]; }
                        if (p + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                        if (p + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                    }
                }
                tile[lid] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 16u; k += 1u) {
                let p = d0 + k * 4u;
                var pv = vec4<f32>(0.0);
                if (p + 3u < dp) {
                    let b = pbase + p;
                    pv = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                } else {
                    let b = pbase + p;
                    if (p < dp) { pv.x = point_keys[b]; }
                    if (p + 1u < dp) { pv.y = point_keys[b + 1u]; }
                    if (p + 2u < dp) { pv.z = point_keys[b + 2u]; }
                }
                for (var j = 0u; j < 8u; j += 1u) {
                    acc[j] += dot(pv, tile[j * 16u + k]);
                }
            }
        }
        for (var j = 0u; j < 8u; j += 1u) {
            let ci = c0 + j;
            if (ci < c && acc[j] > best_sim) {
                best_sim = acc[j];
                best = ci;
            }
        }
    }
    if (valid) {
        assignments[row_off + gid.x] = best;
    }
}
