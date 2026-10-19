@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 128>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = gid.x < n_band;
    let l = min(gid.x, n_band - 1u);
    let prow = l * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        var s4 = 0.0; var s5 = 0.0; var s6 = 0.0; var s7 = 0.0;
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            for (var r = 0u; r < 2u; r += 1u) {
                let t = lid + r * 64u;
                let cj = t / 16u;
                let k = t % 16u;
                let ci = c0 + cj;
                let db = d0 + k * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let base = ci * dp + db;
                    if (db + 0u < dp) { v.x = centroid_keys[base]; }
                    if (db + 1u < dp) { v.y = centroid_keys[base + 1u]; }
                    if (db + 2u < dp) { v.z = centroid_keys[base + 2u]; }
                    if (db + 3u < dp) { v.w = centroid_keys[base + 3u]; }
                }
                tile[t] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 16u; k += 1u) {
                let db = d0 + k * 4u;
                var pv = vec4<f32>(0.0);
                let pb = prow + db;
                if (db + 0u < dp) { pv.x = point_keys[pb]; }
                if (db + 1u < dp) { pv.y = point_keys[pb + 1u]; }
                if (db + 2u < dp) { pv.z = point_keys[pb + 2u]; }
                if (db + 3u < dp) { pv.w = point_keys[pb + 3u]; }
                s0 += dot(pv, tile[k]);
                s1 += dot(pv, tile[16u + k]);
                s2 += dot(pv, tile[32u + k]);
                s3 += dot(pv, tile[48u + k]);
                s4 += dot(pv, tile[64u + k]);
                s5 += dot(pv, tile[80u + k]);
                s6 += dot(pv, tile[96u + k]);
                s7 += dot(pv, tile[112u + k]);
            }
            workgroupBarrier();
        }
        if (c0 + 0u < c && s0 > best_sim) { best_sim = s0; best = c0; }
        if (c0 + 1u < c && s1 > best_sim) { best_sim = s1; best = c0 + 1u; }
        if (c0 + 2u < c && s2 > best_sim) { best_sim = s2; best = c0 + 2u; }
        if (c0 + 3u < c && s3 > best_sim) { best_sim = s3; best = c0 + 3u; }
        if (c0 + 4u < c && s4 > best_sim) { best_sim = s4; best = c0 + 4u; }
        if (c0 + 5u < c && s5 > best_sim) { best_sim = s5; best = c0 + 5u; }
        if (c0 + 6u < c && s6 > best_sim) { best_sim = s6; best = c0 + 6u; }
        if (c0 + 7u < c && s7 > best_sim) { best_sim = s7; best = c0 + 7u; }
    }
    if (valid) {
        assignments[row_off + gid.x] = best;
    }
}
