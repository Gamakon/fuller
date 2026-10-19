@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let local = gid.x;
    let valid = local < n_band;
    var lrow = local;
    if (!valid) { lrow = n_band - 1u; }
    let pbase = lrow * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        var s4 = 0.0; var s5 = 0.0; var s6 = 0.0; var s7 = 0.0;
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var r = 0u; r < 2u; r += 1u) {
                let t = lid + r * 128u;
                let j = t >> 5u;
                let d = d0 + (t & 31u) * 4u;
                let ci = c0 + j;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let b = ci * dp + d;
                    if (d + 3u < dp) {
                        v = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u],
                                      centroid_keys[b + 2u], centroid_keys[b + 3u]);
                    } else {
                        if (d < dp) { v.x = centroid_keys[b]; }
                        if (d + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                        if (d + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                    }
                }
                tile[t] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 32u; k += 1u) {
                let d = d0 + k * 4u;
                if (d < dp) {
                    let b = pbase + d;
                    var pv = vec4<f32>(0.0);
                    if (d + 3u < dp) {
                        pv = vec4<f32>(point_keys[b], point_keys[b + 1u],
                                       point_keys[b + 2u], point_keys[b + 3u]);
                    } else {
                        pv.x = point_keys[b];
                        if (d + 1u < dp) { pv.y = point_keys[b + 1u]; }
                        if (d + 2u < dp) { pv.z = point_keys[b + 2u]; }
                    }
                    s0 += dot(pv, tile[k]);
                    s1 += dot(pv, tile[32u + k]);
                    s2 += dot(pv, tile[64u + k]);
                    s3 += dot(pv, tile[96u + k]);
                    s4 += dot(pv, tile[128u + k]);
                    s5 += dot(pv, tile[160u + k]);
                    s6 += dot(pv, tile[192u + k]);
                    s7 += dot(pv, tile[224u + k]);
                }
            }
        }
        if (c0 < c && s0 > best_sim) { best_sim = s0; best = c0; }
        if (c0 + 1u < c && s1 > best_sim) { best_sim = s1; best = c0 + 1u; }
        if (c0 + 2u < c && s2 > best_sim) { best_sim = s2; best = c0 + 2u; }
        if (c0 + 3u < c && s3 > best_sim) { best_sim = s3; best = c0 + 3u; }
        if (c0 + 4u < c && s4 > best_sim) { best_sim = s4; best = c0 + 4u; }
        if (c0 + 5u < c && s5 > best_sim) { best_sim = s5; best = c0 + 5u; }
        if (c0 + 6u < c && s6 > best_sim) { best_sim = s6; best = c0 + 6u; }
        if (c0 + 7u < c && s7 > best_sim) { best_sim = s7; best = c0 + 7u; }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
