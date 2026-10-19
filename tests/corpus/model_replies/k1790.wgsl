@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let local = gid.x;
    let valid = local < n_band;
    var pl = local;
    if (!valid) { pl = n_band - 1u; }
    let pbase = pl * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cg: u32 = 0u; cg < c; cg += 16u) {
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
        var a8 = 0.0; var a9 = 0.0; var a10 = 0.0; var a11 = 0.0;
        var a12 = 0.0; var a13 = 0.0; var a14 = 0.0; var a15 = 0.0;
        for (var d0: u32 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            for (var i: u32 = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx >> 6u;
                let k = idx & 63u;
                var v = 0.0;
                if (cg + cc < c && d0 + k < dp) {
                    v = centroid_keys[(cg + cc) * dp + d0 + k];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var k: u32 = 0u; k < 64u; k += 1u) {
                if (d0 + k < dp) {
                    let pv = point_keys[pbase + d0 + k];
                    a0 += pv * tile[k];
                    a1 += pv * tile[64u + k];
                    a2 += pv * tile[128u + k];
                    a3 += pv * tile[192u + k];
                    a4 += pv * tile[256u + k];
                    a5 += pv * tile[320u + k];
                    a6 += pv * tile[384u + k];
                    a7 += pv * tile[448u + k];
                    a8 += pv * tile[512u + k];
                    a9 += pv * tile[576u + k];
                    a10 += pv * tile[640u + k];
                    a11 += pv * tile[704u + k];
                    a12 += pv * tile[768u + k];
                    a13 += pv * tile[832u + k];
                    a14 += pv * tile[896u + k];
                    a15 += pv * tile[960u + k];
                }
            }
        }
        if (cg + 0u < c && a0 > best_sim) { best_sim = a0; best = cg + 0u; }
        if (cg + 1u < c && a1 > best_sim) { best_sim = a1; best = cg + 1u; }
        if (cg + 2u < c && a2 > best_sim) { best_sim = a2; best = cg + 2u; }
        if (cg + 3u < c && a3 > best_sim) { best_sim = a3; best = cg + 3u; }
        if (cg + 4u < c && a4 > best_sim) { best_sim = a4; best = cg + 4u; }
        if (cg + 5u < c && a5 > best_sim) { best_sim = a5; best = cg + 5u; }
        if (cg + 6u < c && a6 > best_sim) { best_sim = a6; best = cg + 6u; }
        if (cg + 7u < c && a7 > best_sim) { best_sim = a7; best = cg + 7u; }
        if (cg + 8u < c && a8 > best_sim) { best_sim = a8; best = cg + 8u; }
        if (cg + 9u < c && a9 > best_sim) { best_sim = a9; best = cg + 9u; }
        if (cg + 10u < c && a10 > best_sim) { best_sim = a10; best = cg + 10u; }
        if (cg + 11u < c && a11 > best_sim) { best_sim = a11; best = cg + 11u; }
        if (cg + 12u < c && a12 > best_sim) { best_sim = a12; best = cg + 12u; }
        if (cg + 13u < c && a13 > best_sim) { best_sim = a13; best = cg + 13u; }
        if (cg + 14u < c && a14 > best_sim) { best_sim = a14; best = cg + 14u; }
        if (cg + 15u < c && a15 > best_sim) { best_sim = a15; best = cg + 15u; }
    }

    if (valid) {
        assignments[row_off + local] = best;
    }
}
