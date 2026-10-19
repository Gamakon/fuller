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
    let pl = min(local, n_band - 1u);
    let pbase = pl * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var ci0 = 0u; ci0 < c; ci0 += 8u) {
        var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        var s4 = 0.0; var s5 = 0.0; var s6 = 0.0; var s7 = 0.0;
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            for (var i = 0u; i < 1024u; i += 64u) {
                let idx = i + lid;
                let j = idx / 128u;
                let p = idx % 128u;
                let ci = ci0 + j;
                let d = d0 + p;
                var v = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var q = 0u; q < 128u; q += 4u) {
                let d = d0 + q;
                var pv = vec4<f32>(0.0);
                if (d + 3u < dp) {
                    let b = pbase + d;
                    pv = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                } else {
                    if (d < dp) { pv.x = point_keys[pbase + d]; }
                    if (d + 1u < dp) { pv.y = point_keys[pbase + d + 1u]; }
                    if (d + 2u < dp) { pv.z = point_keys[pbase + d + 2u]; }
                    if (d + 3u < dp) { pv.w = point_keys[pbase + d + 3u]; }
                }
                s0 += dot(pv, vec4<f32>(tile[q], tile[q + 1u], tile[q + 2u], tile[q + 3u]));
                s1 += dot(pv, vec4<f32>(tile[128u + q], tile[128u + q + 1u], tile[128u + q + 2u], tile[128u + q + 3u]));
                s2 += dot(pv, vec4<f32>(tile[256u + q], tile[256u + q + 1u], tile[256u + q + 2u], tile[256u + q + 3u]));
                s3 += dot(pv, vec4<f32>(tile[384u + q], tile[384u + q + 1u], tile[384u + q + 2u], tile[384u + q + 3u]));
                s4 += dot(pv, vec4<f32>(tile[512u + q], tile[512u + q + 1u], tile[512u + q + 2u], tile[512u + q + 3u]));
                s5 += dot(pv, vec4<f32>(tile[640u + q], tile[640u + q + 1u], tile[640u + q + 2u], tile[640u + q + 3u]));
                s6 += dot(pv, vec4<f32>(tile[768u + q], tile[768u + q + 1u], tile[768u + q + 2u], tile[768u + q + 3u]));
                s7 += dot(pv, vec4<f32>(tile[896u + q], tile[896u + q + 1u], tile[896u + q + 2u], tile[896u + q + 3u]));
            }
            workgroupBarrier();
        }
        if (ci0 < c && s0 > best_sim) { best_sim = s0; best = ci0; }
        if (ci0 + 1u < c && s1 > best_sim) { best_sim = s1; best = ci0 + 1u; }
        if (ci0 + 2u < c && s2 > best_sim) { best_sim = s2; best = ci0 + 2u; }
        if (ci0 + 3u < c && s3 > best_sim) { best_sim = s3; best = ci0 + 3u; }
        if (ci0 + 4u < c && s4 > best_sim) { best_sim = s4; best = ci0 + 4u; }
        if (ci0 + 5u < c && s5 > best_sim) { best_sim = s5; best = ci0 + 5u; }
        if (ci0 + 6u < c && s6 > best_sim) { best_sim = s6; best = ci0 + 6u; }
        if (ci0 + 7u < c && s7 > best_sim) { best_sim = s7; best = ci0 + 7u; }
    }

    if (valid) {
        assignments[row_off + local] = best;
    }
}
