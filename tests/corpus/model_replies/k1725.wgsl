@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let active = local < n_band;
    let pbase = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        var s2: f32 = 0.0;
        var s3: f32 = 0.0;
        var s4: f32 = 0.0;
        var s5: f32 = 0.0;
        var s6: f32 = 0.0;
        var s7: f32 = 0.0;
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 4u; i += 1u) {
                let s = i * 64u + lid;
                let cc = s / 32u;
                let q = s % 32u;
                let ci = c0 + cc;
                let d = d0 + q * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let b = ci * dp + d;
                    if (d < dp) { v.x = centroid_keys[b]; }
                    if (d + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                    if (d + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                    if (d + 3u < dp) { v.w = centroid_keys[b + 3u]; }
                }
                tile[s] = v;
            }
            workgroupBarrier();
            if (active) {
                for (var q = 0u; q < 32u; q += 1u) {
                    let d = d0 + q * 4u;
                    var pv = vec4<f32>(0.0);
                    if (d + 3u < dp) {
                        let b = pbase + d;
                        pv = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                    } else if (d < dp) {
                        let b = pbase + d;
                        pv.x = point_keys[b];
                        if (d + 1u < dp) { pv.y = point_keys[b + 1u]; }
                        if (d + 2u < dp) { pv.z = point_keys[b + 2u]; }
                    }
                    s0 += dot(pv, tile[q]);
                    s1 += dot(pv, tile[32u + q]);
                    s2 += dot(pv, tile[64u + q]);
                    s3 += dot(pv, tile[96u + q]);
                    s4 += dot(pv, tile[128u + q]);
                    s5 += dot(pv, tile[160u + q]);
                    s6 += dot(pv, tile[192u + q]);
                    s7 += dot(pv, tile[224u + q]);
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
    if (active) {
        assignments[row_off + local] = best;
    }
}
