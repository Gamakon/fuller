@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let l = gid.x;
    let valid = l < n_band;
    let lp = min(l, n_band - 1u);
    let base = lp * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cg = 0u; cg < c; cg += 8u) {
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            let dd = d0 + lid;
            for (var i = 0u; i < 8u; i += 1u) {
                var v = 0.0;
                if (cg + i < c && dd < dp) {
                    v = centroid_keys[(cg + i) * dp + dd];
                }
                tile[i * 128u + lid] = v;
            }
            workgroupBarrier();
            let kn = min(128u, dp - d0);
            for (var k = 0u; k < 128u; k += 1u) {
                if (k < kn) {
                    let p = point_keys[base + d0 + k];
                    a0 += p * tile[k];
                    a1 += p * tile[128u + k];
                    a2 += p * tile[256u + k];
                    a3 += p * tile[384u + k];
                    a4 += p * tile[512u + k];
                    a5 += p * tile[640u + k];
                    a6 += p * tile[768u + k];
                    a7 += p * tile[896u + k];
                }
            }
        }
        if (cg + 0u < c && a0 > best_sim) { best_sim = a0; best = cg; }
        if (cg + 1u < c && a1 > best_sim) { best_sim = a1; best = cg + 1u; }
        if (cg + 2u < c && a2 > best_sim) { best_sim = a2; best = cg + 2u; }
        if (cg + 3u < c && a3 > best_sim) { best_sim = a3; best = cg + 3u; }
        if (cg + 4u < c && a4 > best_sim) { best_sim = a4; best = cg + 4u; }
        if (cg + 5u < c && a5 > best_sim) { best_sim = a5; best = cg + 5u; }
        if (cg + 6u < c && a6 > best_sim) { best_sim = a6; best = cg + 6u; }
        if (cg + 7u < c && a7 > best_sim) { best_sim = a7; best = cg + 7u; }
    }

    if (valid) {
        assignments[row_off + l] = best;
    }
}
