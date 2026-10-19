@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let l = gid.x;
    let pl = min(l, n_band - 1u);
    let pbase = pl * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb = 0u; cb < c; cb += 16u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            for (var i = 0u; i < 4u; i += 1u) {
                let e = lid + i * 64u;
                let p = e >> 2u;
                let g = e & 3u;
                let d = d0 + p;
                var v = vec4<f32>(0.0);
                if (d < dp) {
                    for (var k = 0u; k < 4u; k += 1u) {
                        let ci = cb + g * 4u + k;
                        if (ci < c) {
                            v[k] = centroid_keys[ci * dp + d];
                        }
                    }
                }
                tile[e] = v;
            }
            workgroupBarrier();
            for (var p = 0u; p < 64u; p += 1u) {
                let d = d0 + p;
                var x = 0.0;
                if (d < dp) {
                    x = point_keys[pbase + d];
                }
                a0 = a0 + x * tile[p * 4u];
                a1 = a1 + x * tile[p * 4u + 1u];
                a2 = a2 + x * tile[p * 4u + 2u];
                a3 = a3 + x * tile[p * 4u + 3u];
            }
            workgroupBarrier();
        }
        var arr = array<vec4<f32>, 4>(a0, a1, a2, a3);
        for (var j = 0u; j < 16u; j += 1u) {
            let ci = cb + j;
            if (ci < c) {
                let s = arr[j >> 2u][j & 3u];
                if (s > best_sim) {
                    best_sim = s;
                    best = ci;
                }
            }
        }
    }

    if (l < n_band) {
        assignments[row_off + l] = best;
    }
}
