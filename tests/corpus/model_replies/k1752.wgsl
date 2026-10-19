@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let l = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = l < n_band;
    var pbase = 0u;
    if (valid) { pbase = l * dp; }

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        var acc: array<f32, 16>;
        for (var j = 0u; j < 16u; j += 1u) { acc[j] = 0.0; }
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            // cooperative load
            for (var i = 0u; i < 2u; i += 1u) {
                let idx = i * 128u + lid;
                let cj = idx >> 4u;
                let v = idx & 15u;
                let cc = c0 + cj;
                let d = d0 + v * 4u;
                var t = vec4<f32>(0.0);
                if (cc < c) {
                    let b = cc * dp + d;
                    if (d + 3u < dp) {
                        t = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
                    } else {
                        if (d < dp) { t.x = centroid_keys[b]; }
                        if (d + 1u < dp) { t.y = centroid_keys[b + 1u]; }
                        if (d + 2u < dp) { t.z = centroid_keys[b + 2u]; }
                    }
                }
                tile[idx] = t;
            }
            workgroupBarrier();
            if (valid) {
                for (var k = 0u; k < 16u; k += 1u) {
                    let d = d0 + k * 4u;
                    var pv = vec4<f32>(0.0);
                    if (d + 3u < dp) {
                        let b = pbase + d;
                        pv = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                    } else {
                        if (d < dp) { pv.x = point_keys[pbase + d]; }
                        if (d + 1u < dp) { pv.y = point_keys[pbase + d + 1u]; }
                        if (d + 2u < dp) { pv.z = point_keys[pbase + d + 2u]; }
                    }
                    for (var j = 0u; j < 16u; j += 1u) {
                        acc[j] = acc[j] + dot(pv, tile[j * 16u + k]);
                    }
                }
            }
            workgroupBarrier();
        }
        for (var j = 0u; j < 16u; j += 1u) {
            if (c0 + j < c) {
                if (acc[j] > best_sim) { best_sim = acc[j]; best = c0 + j; }
            }
        }
    }
    if (valid) {
        assignments[row_off + l] = best;
    }
}
