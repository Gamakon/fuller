@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let local = gid.x;
    let lp = min(local, n_band - 1u);
    let base = lp * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    for (var cb: u32 = 0u; cb < c; cb += 8u) {
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i: u32 = 0u; i < 4u; i += 1u) {
                let idx = i * 64u + lid;
                let k = idx / 32u;
                let q = idx % 32u;
                let ci = cb + k;
                let p = d0 + q * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let o = ci * dp + p;
                    if (p < dp) { v.x = centroid_keys[o]; }
                    if (p + 1u < dp) { v.y = centroid_keys[o + 1u]; }
                    if (p + 2u < dp) { v.z = centroid_keys[o + 2u]; }
                    if (p + 3u < dp) { v.w = centroid_keys[o + 3u]; }
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var q: u32 = 0u; q < 32u; q += 1u) {
                let p = d0 + q * 4u;
                var pv = vec4<f32>(0.0);
                if (p < dp) { pv.x = point_keys[base + p]; }
                if (p + 1u < dp) { pv.y = point_keys[base + p + 1u]; }
                if (p + 2u < dp) { pv.z = point_keys[base + p + 2u]; }
                if (p + 3u < dp) { pv.w = point_keys[base + p + 3u]; }
                a0 += dot(pv, tile[q]);
                a1 += dot(pv, tile[32u + q]);
                a2 += dot(pv, tile[64u + q]);
                a3 += dot(pv, tile[96u + q]);
                a4 += dot(pv, tile[128u + q]);
                a5 += dot(pv, tile[160u + q]);
                a6 += dot(pv, tile[192u + q]);
                a7 += dot(pv, tile[224u + q]);
            }
        }
        if (cb < c && a0 > best_sim) { best_sim = a0; best = cb; }
        if (cb + 1u < c && a1 > best_sim) { best_sim = a1; best = cb + 1u; }
        if (cb + 2u < c && a2 > best_sim) { best_sim = a2; best = cb + 2u; }
        if (cb + 3u < c && a3 > best_sim) { best_sim = a3; best = cb + 3u; }
        if (cb + 4u < c && a4 > best_sim) { best_sim = a4; best = cb + 4u; }
        if (cb + 5u < c && a5 > best_sim) { best_sim = a5; best = cb + 5u; }
        if (cb + 6u < c && a6 > best_sim) { best_sim = a6; best = cb + 6u; }
        if (cb + 7u < c && a7 > best_sim) { best_sim = a7; best = cb + 7u; }
    }
    if (local < n_band) {
        assignments[row_off + local] = best;
    }
}
