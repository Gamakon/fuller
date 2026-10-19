@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 512>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let l = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = l < n_band;
    let base = l * dp;
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
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            for (var k = 0u; k < 4u; k += 1u) {
                let idx = lid + k * 128u;
                let ci = idx >> 6u;
                let p = idx & 63u;
                let cc = c0 + ci;
                let dd = d0 + p;
                var v: f32 = 0.0;
                if (cc < c && dd < dp) {
                    v = centroid_keys[cc * dp + dd];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var p = 0u; p < 64u; p += 1u) {
                    let dd = d0 + p;
                    if (dd < dp) {
                        let x = point_keys[base + dd];
                        s0 = s0 + x * tile[p];
                        s1 = s1 + x * tile[64u + p];
                        s2 = s2 + x * tile[128u + p];
                        s3 = s3 + x * tile[192u + p];
                        s4 = s4 + x * tile[256u + p];
                        s5 = s5 + x * tile[320u + p];
                        s6 = s6 + x * tile[384u + p];
                        s7 = s7 + x * tile[448u + p];
                    }
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
        assignments[row_off + l] = best;
    }
}
