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
    let base = select(0u, local * dp, valid);
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    for (var c0: u32 = 0u; c0 < c; c0 += 8u) {
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var a4: f32 = 0.0;
        var a5: f32 = 0.0;
        var a6: f32 = 0.0;
        var a7: f32 = 0.0;
        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var k: u32 = 0u; k < 16u; k += 1u) {
                let idx = k * 64u + lid;
                let cj = idx >> 7u;
                let pp = idx & 127u;
                var v: f32 = 0.0;
                if (c0 + cj < c && d0 + pp < dp) {
                    v = centroid_keys[(c0 + cj) * dp + d0 + pp];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var p: u32 = 0u; p < 128u; p += 1u) {
                var pv: f32 = 0.0;
                if (d0 + p < dp) {
                    pv = point_keys[base + d0 + p];
                }
                a0 = a0 + pv * tile[p];
                a1 = a1 + pv * tile[128u + p];
                a2 = a2 + pv * tile[256u + p];
                a3 = a3 + pv * tile[384u + p];
                a4 = a4 + pv * tile[512u + p];
                a5 = a5 + pv * tile[640u + p];
                a6 = a6 + pv * tile[768u + p];
                a7 = a7 + pv * tile[896u + p];
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
