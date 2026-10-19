@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let local = gid.x;
    let valid = local < n_band;
    var lrow = 0u;
    if (valid) { lrow = local; }
    let pb = lrow * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var s: array<f32, 8>;
        for (var j = 0u; j < 8u; j += 1u) { s[j] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var k = 0u; k < 8u; k += 1u) {
                let idx = k * 128u + lidx;
                let cj = idx >> 7u;
                let p = idx & 127u;
                let cg = c0 + cj;
                var v = 0.0;
                if (cg < c && d0 + p < dp) {
                    v = centroid_keys[cg * dp + d0 + p];
                }
                tile[idx] = v;
            }
            workgroupBarrier();

            if (d0 + 128u <= dp) {
                for (var p = 0u; p < 128u; p += 4u) {
                    let a0 = point_keys[pb + d0 + p];
                    let a1 = point_keys[pb + d0 + p + 1u];
                    let a2 = point_keys[pb + d0 + p + 2u];
                    let a3 = point_keys[pb + d0 + p + 3u];
                    for (var j = 0u; j < 8u; j += 1u) {
                        let t = j * 128u + p;
                        s[j] = s[j] + a0 * tile[t] + a1 * tile[t + 1u] + a2 * tile[t + 2u] + a3 * tile[t + 3u];
                    }
                }
            } else {
                for (var p = 0u; p < 128u; p += 1u) {
                    if (d0 + p < dp) {
                        let a = point_keys[pb + d0 + p];
                        for (var j = 0u; j < 8u; j += 1u) {
                            s[j] = s[j] + a * tile[j * 128u + p];
                        }
                    }
                }
            }
        }

        for (var j = 0u; j < 8u; j += 1u) {
            let ci = c0 + j;
            if (ci < c && s[j] > best_sim) {
                best_sim = s[j];
                best = ci;
            }
        }
    }

    if (valid) {
        assignments[row_off + local] = best;
    }
}
