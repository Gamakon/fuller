@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = local < n_band;
    let pbase = local * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    let lj = t / 32u;
    let lv = t % 32u;

    for (var cb: u32 = 0u; cb < c; cb += 8u) {
        var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
        var a4: f32 = 0.0; var a5: f32 = 0.0; var a6: f32 = 0.0; var a7: f32 = 0.0;
        for (var dt: u32 = 0u; dt < dp; dt += 128u) {
            workgroupBarrier();
            {
                let ci = cb + lj;
                let d = dt + lv * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let b = ci * dp + d;
                    if (d < dp) { v.x = centroid_keys[b]; }
                    if (d + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                    if (d + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                    if (d + 3u < dp) { v.w = centroid_keys[b + 3u]; }
                }
                tile[t] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var q: u32 = 0u; q < 32u; q += 1u) {
                    let d = dt + q * 4u;
                    var pv = vec4<f32>(0.0);
                    if (d < dp) { pv.x = point_keys[pbase + d]; }
                    if (d + 1u < dp) { pv.y = point_keys[pbase + d + 1u]; }
                    if (d + 2u < dp) { pv.z = point_keys[pbase + d + 2u]; }
                    if (d + 3u < dp) { pv.w = point_keys[pbase + d + 3u]; }
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

    if (valid) {
        assignments[row_off + local] = best;
    }
}
