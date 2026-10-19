@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

fn load_pt(base: u32, d: u32, dp: u32) -> vec4<f32> {
    var r = vec4<f32>(0.0);
    if (d + 3u < dp) {
        r = vec4<f32>(point_keys[base + d], point_keys[base + d + 1u], point_keys[base + d + 2u], point_keys[base + d + 3u]);
    } else {
        if (d < dp) { r.x = point_keys[base + d]; }
        if (d + 1u < dp) { r.y = point_keys[base + d + 1u]; }
        if (d + 2u < dp) { r.z = point_keys[base + d + 2u]; }
    }
    return r;
}

fn load_ct(base: u32, d: u32, dp: u32) -> vec4<f32> {
    var r = vec4<f32>(0.0);
    if (d + 3u < dp) {
        r = vec4<f32>(centroid_keys[base + d], centroid_keys[base + d + 1u], centroid_keys[base + d + 2u], centroid_keys[base + d + 3u]);
    } else {
        if (d < dp) { r.x = centroid_keys[base + d]; }
        if (d + 1u < dp) { r.y = centroid_keys[base + d + 1u]; }
        if (d + 2u < dp) { r.z = centroid_keys[base + d + 2u]; }
    }
    return r;
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let l = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = l < n_band;
    let pbase = l * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cg: u32 = 0u; cg < c; cg += 8u) {
        var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        var s4 = 0.0; var s5 = 0.0; var s6 = 0.0; var s7 = 0.0;
        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            for (var i: u32 = 0u; i < 256u; i += 128u) {
                let j = i + lid;
                let cj = j / 32u;
                let v = j % 32u;
                let ci = cg + cj;
                let d = d0 + v * 4u;
                var t = vec4<f32>(0.0);
                if (ci < c) { t = load_ct(ci * dp, d, dp); }
                tile[j] = t;
            }
            workgroupBarrier();
            if (valid) {
                for (var v: u32 = 0u; v < 32u; v += 1u) {
                    let d = d0 + v * 4u;
                    if (d < dp) {
                        let pv = load_pt(pbase, d, dp);
                        s0 += dot(pv, tile[v]);
                        s1 += dot(pv, tile[32u + v]);
                        s2 += dot(pv, tile[64u + v]);
                        s3 += dot(pv, tile[96u + v]);
                        s4 += dot(pv, tile[128u + v]);
                        s5 += dot(pv, tile[160u + v]);
                        s6 += dot(pv, tile[192u + v]);
                        s7 += dot(pv, tile[224u + v]);
                    }
                }
            }
            workgroupBarrier();
        }
        if (cg < c && s0 > best_sim) { best_sim = s0; best = cg; }
        if (cg + 1u < c && s1 > best_sim) { best_sim = s1; best = cg + 1u; }
        if (cg + 2u < c && s2 > best_sim) { best_sim = s2; best = cg + 2u; }
        if (cg + 3u < c && s3 > best_sim) { best_sim = s3; best = cg + 3u; }
        if (cg + 4u < c && s4 > best_sim) { best_sim = s4; best = cg + 4u; }
        if (cg + 5u < c && s5 > best_sim) { best_sim = s5; best = cg + 5u; }
        if (cg + 6u < c && s6 > best_sim) { best_sim = s6; best = cg + 6u; }
        if (cg + 7u < c && s7 > best_sim) { best_sim = s7; best = cg + 7u; }
    }
    if (valid) {
        assignments[row_off + l] = best;
    }
}
