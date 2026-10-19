@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 512>;

fn ldp(base: u32, p: u32, dp: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    if (p + 3u < dp) {
        v = vec4<f32>(point_keys[base + p], point_keys[base + p + 1u], point_keys[base + p + 2u], point_keys[base + p + 3u]);
    } else if (p < dp) {
        for (var k = 0u; k < 4u; k += 1u) {
            if (p + k < dp) { v[k] = point_keys[base + p + k]; }
        }
    }
    return v;
}

fn ldc(base: u32, p: u32, dp: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    if (p + 3u < dp) {
        v = vec4<f32>(centroid_keys[base + p], centroid_keys[base + p + 1u], centroid_keys[base + p + 2u], centroid_keys[base + p + 3u]);
    } else if (p < dp) {
        for (var k = 0u; k < 4u; k += 1u) {
            if (p + k < dp) { v[k] = centroid_keys[base + p + k]; }
        }
    }
    return v;
}

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

    for (var cg = 0u; cg < c; cg += 8u) {
        var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        var s4 = 0.0; var s5 = 0.0; var s6 = 0.0; var s7 = 0.0;
        for (var d0 = 0u; d0 < dp; d0 += 256u) {
            workgroupBarrier();
            for (var k = 0u; k < 512u; k += 64u) {
                let idx = k + lid;
                let cj = idx / 64u;
                let q = idx % 64u;
                let ci = cg + cj;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    v = ldc(ci * dp, d0 + q * 4u, dp);
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var q = 0u; q < 64u; q += 1u) {
                    let pv = ldp(pbase, d0 + q * 4u, dp);
                    s0 += dot(pv, tile[q]);
                    s1 += dot(pv, tile[64u + q]);
                    s2 += dot(pv, tile[128u + q]);
                    s3 += dot(pv, tile[192u + q]);
                    s4 += dot(pv, tile[256u + q]);
                    s5 += dot(pv, tile[320u + q]);
                    s6 += dot(pv, tile[384u + q]);
                    s7 += dot(pv, tile[448u + q]);
                }
            }
        }
        if (cg + 0u < c && s0 > best_sim) { best_sim = s0; best = cg; }
        if (cg + 1u < c && s1 > best_sim) { best_sim = s1; best = cg + 1u; }
        if (cg + 2u < c && s2 > best_sim) { best_sim = s2; best = cg + 2u; }
        if (cg + 3u < c && s3 > best_sim) { best_sim = s3; best = cg + 3u; }
        if (cg + 4u < c && s4 > best_sim) { best_sim = s4; best = cg + 4u; }
        if (cg + 5u < c && s5 > best_sim) { best_sim = s5; best = cg + 5u; }
        if (cg + 6u < c && s6 > best_sim) { best_sim = s6; best = cg + 6u; }
        if (cg + 7u < c && s7 > best_sim) { best_sim = s7; best = cg + 7u; }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
