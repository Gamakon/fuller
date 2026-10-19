@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 512>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let basep = wid.x * 256u;

    var rowbase: array<u32, 4>;
    for (var j = 0u; j < 4u; j += 1u) {
        let l = basep + j * 64u + lid;
        rowbase[j] = min(l, n - 1u) * dp;
    }
    var best: array<u32, 4>;
    var best_sim: array<f32, 4>;
    for (var j = 0u; j < 4u; j += 1u) {
        best[j] = 0u;
        best_sim[j] = -3.4e38;
    }

    for (var t = 0u; t < c; t += 8u) {
        var acc: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) {
            acc[i] = 0.0;
        }
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            for (var i = 0u; i < 8u; i += 1u) {
                let ci = t + i;
                let d = d0 + lid;
                var v = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                tile[i * 64u + lid] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 16u; k += 1u) {
                let d = min(d0 + k * 4u, dp - 4u);
                var pv: array<vec4<f32>, 4>;
                for (var j = 0u; j < 4u; j += 1u) {
                    let b = rowbase[j] + d;
                    pv[j] = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                }
                for (var m = 0u; m < 8u; m += 1u) {
                    let o = m * 64u + k * 4u;
                    let cv = vec4<f32>(tile[o], tile[o + 1u], tile[o + 2u], tile[o + 3u]);
                    for (var j = 0u; j < 4u; j += 1u) {
                        acc[j * 8u + m] += dot(pv[j], cv);
                    }
                }
            }
        }
        for (var m = 0u; m < 8u; m += 1u) {
            let ci = t + m;
            for (var j = 0u; j < 4u; j += 1u) {
                let s = acc[j * 8u + m];
                if (ci < c && s > best_sim[j]) {
                    best_sim[j] = s;
                    best[j] = ci;
                }
            }
        }
    }

    for (var j = 0u; j < 4u; j += 1u) {
        let l = basep + j * 64u + lid;
        if (l < n) {
            assignments[row_off + l] = best[j];
        }
    }
}
