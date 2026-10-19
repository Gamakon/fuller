@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 512>;

fn pk(idx: u32, d: u32, dp: u32, base: u32) -> f32 {
    if (d < dp) { return point_keys[base + d]; }
    return 0.0;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let local = gid.x;
    let valid = local < n_band;
    var lp = local;
    if (!valid) { lp = 0u; }
    let base = lp * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 16u) {
        var acc: array<f32, 16>;
        for (var j = 0u; j < 16u; j++) { acc[j] = 0.0; }

        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 2u; i++) {
                let e = i * 256u + lid;
                let cj = e / 32u;
                let v = e % 32u;
                let ci = cb + cj;
                let dd = d0 + v * 4u;
                var t = vec4<f32>(0.0);
                if (ci < c) {
                    let cbase = ci * dp;
                    if (dd + 3u < dp) {
                        t = vec4<f32>(centroid_keys[cbase + dd], centroid_keys[cbase + dd + 1u],
                                      centroid_keys[cbase + dd + 2u], centroid_keys[cbase + dd + 3u]);
                    } else {
                        if (dd < dp) { t.x = centroid_keys[cbase + dd]; }
                        if (dd + 1u < dp) { t.y = centroid_keys[cbase + dd + 1u]; }
                        if (dd + 2u < dp) { t.z = centroid_keys[cbase + dd + 2u]; }
                    }
                }
                tile[e] = t;
            }
            workgroupBarrier();

            for (var q = 0u; q < 32u; q++) {
                let d = d0 + q * 4u;
                let pv = vec4<f32>(pk(0u, d, dp, base), pk(0u, d + 1u, dp, base),
                                   pk(0u, d + 2u, dp, base), pk(0u, d + 3u, dp, base));
                for (var j = 0u; j < 16u; j++) {
                    acc[j] += dot(pv, tile[j * 32u + q]);
                }
            }
        }

        for (var j = 0u; j < 16u; j++) {
            let ci = cb + j;
            if (ci < c && acc[j] > best_sim) {
                best_sim = acc[j];
                best = ci;
            }
        }
    }

    if (valid) {
        assignments[row_off + local] = best;
    }
}
