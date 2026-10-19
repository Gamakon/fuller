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
    let pbase = local * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 16u) {
        var acc: array<f32, 16>;
        for (var k = 0u; k < 16u; k += 1u) { acc[k] = 0.0; }

        for (var db: u32 = 0u; db < dp; db += 64u) {
            workgroupBarrier();
            for (var i = 0u; i < 8u; i += 1u) {
                let idx = i * 128u + lidx;
                let cc = idx >> 6u;
                let d = idx & 63u;
                var v: f32 = 0.0;
                if (cb + cc < c && db + d < dp) {
                    v = centroid_keys[(cb + cc) * dp + db + d];
                }
                tile[idx] = v;
            }
            workgroupBarrier();

            if (valid) {
                let lim = min(64u, dp - db);
                for (var d = 0u; d < 64u; d += 1u) {
                    if (d < lim) {
                        let pv = point_keys[pbase + db + d];
                        for (var k = 0u; k < 16u; k += 1u) {
                            acc[k] = acc[k] + pv * tile[k * 64u + d];
                        }
                    }
                }
            }
        }

        for (var k = 0u; k < 16u; k += 1u) {
            if (cb + k < c) {
                if (acc[k] > best_sim) { best_sim = acc[k]; best = cb + k; }
            }
        }
    }

    if (valid) {
        assignments[row_off + local] = best;
    }
}
