@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let l = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = l < n_band;
    let base = l * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    let nchunks = (dp + 127u) / 128u;
    for (var cb: u32 = 0u; cb < c; cb += 8u) {
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        var s2: f32 = 0.0;
        var s3: f32 = 0.0;
        var s4: f32 = 0.0;
        var s5: f32 = 0.0;
        var s6: f32 = 0.0;
        var s7: f32 = 0.0;
        for (var ch: u32 = 0u; ch < nchunks; ch += 1u) {
            let pc = ch * 128u;
            workgroupBarrier();
            for (var i: u32 = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + t;
                let j = idx >> 7u;
                let d = idx & 127u;
                let ci = cb + j;
                let p = pc + d;
                var v: f32 = 0.0;
                if (ci < c && p < dp) {
                    v = centroid_keys[ci * dp + p];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var d: u32 = 0u; d < 128u; d += 1u) {
                    let p = pc + d;
                    if (p < dp) {
                        let pv = point_keys[base + p];
                        s0 = s0 + pv * tile[d];
                        s1 = s1 + pv * tile[128u + d];
                        s2 = s2 + pv * tile[256u + d];
                        s3 = s3 + pv * tile[384u + d];
                        s4 = s4 + pv * tile[512u + d];
                        s5 = s5 + pv * tile[640u + d];
                        s6 = s6 + pv * tile[768u + d];
                        s7 = s7 + pv * tile[896u + d];
                    }
                }
            }
        }
        if (cb < c && s0 > best_sim) { best_sim = s0; best = cb; }
        if (cb + 1u < c && s1 > best_sim) { best_sim = s1; best = cb + 1u; }
        if (cb + 2u < c && s2 > best_sim) { best_sim = s2; best = cb + 2u; }
        if (cb + 3u < c && s3 > best_sim) { best_sim = s3; best = cb + 3u; }
        if (cb + 4u < c && s4 > best_sim) { best_sim = s4; best = cb + 4u; }
        if (cb + 5u < c && s5 > best_sim) { best_sim = s5; best = cb + 5u; }
        if (cb + 6u < c && s6 > best_sim) { best_sim = s6; best = cb + 6u; }
        if (cb + 7u < c && s7 > best_sim) { best_sim = s7; best = cb + 7u; }
    }
    if (valid) {
        assignments[row_off + l] = best;
    }
}
