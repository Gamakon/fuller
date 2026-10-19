@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> sh: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = local < n_band;
    let pbase = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var a4: f32 = 0.0;
        var a5: f32 = 0.0;
        var a6: f32 = 0.0;
        var a7: f32 = 0.0;
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 4u; i += 1u) {
                let j = i * 64u + lid;
                let cc = j >> 5u;
                let q = j & 31u;
                let ci = c0 + cc;
                let d = d0 + q * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c && d < dp) {
                    let b = ci * dp + d;
                    if (d + 3u < dp) {
                        v = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
                    } else {
                        v.x = centroid_keys[b];
                        if (d + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                        if (d + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                    }
                }
                sh[j] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var q = 0u; q < 32u; q += 1u) {
                    let d = d0 + q * 4u;
                    if (d < dp) {
                        let b = pbase + d;
                        var pv = vec4<f32>(0.0);
                        if (d + 3u < dp) {
                            pv = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                        } else {
                            pv.x = point_keys[b];
                            if (d + 1u < dp) { pv.y = point_keys[b + 1u]; }
                            if (d + 2u < dp) { pv.z = point_keys[b + 2u]; }
                        }
                        a0 += dot(pv, sh[q]);
                        a1 += dot(pv, sh[32u + q]);
                        a2 += dot(pv, sh[64u + q]);
                        a3 += dot(pv, sh[96u + q]);
                        a4 += dot(pv, sh[128u + q]);
                        a5 += dot(pv, sh[160u + q]);
                        a6 += dot(pv, sh[192u + q]);
                        a7 += dot(pv, sh[224u + q]);
                    }
                }
            }
        }
        if (c0 < c && a0 > best_sim) { best_sim = a0; best = c0; }
        if (c0 + 1u < c && a1 > best_sim) { best_sim = a1; best = c0 + 1u; }
        if (c0 + 2u < c && a2 > best_sim) { best_sim = a2; best = c0 + 2u; }
        if (c0 + 3u < c && a3 > best_sim) { best_sim = a3; best = c0 + 3u; }
        if (c0 + 4u < c && a4 > best_sim) { best_sim = a4; best = c0 + 4u; }
        if (c0 + 5u < c && a5 > best_sim) { best_sim = a5; best = c0 + 5u; }
        if (c0 + 6u < c && a6 > best_sim) { best_sim = a6; best = c0 + 6u; }
        if (c0 + 7u < c && a7 > best_sim) { best_sim = a7; best = c0 + 7u; }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
