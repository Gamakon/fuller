@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let active = local < n_band;
    let base = local * dp;
    let t = lid.x;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0: u32 = 0u; c0 < c; c0 += 8u) {
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        var s2: f32 = 0.0;
        var s3: f32 = 0.0;
        var s4: f32 = 0.0;
        var s5: f32 = 0.0;
        var s6: f32 = 0.0;
        var s7: f32 = 0.0;
        for (var d0: u32 = 0u; d0 < dp; d0 += 256u) {
            workgroupBarrier();
            for (var i: u32 = 0u; i < 32u; i += 1u) {
                let j = i * 64u + t;
                let cc = j >> 8u;
                let dd = j & 255u;
                let ci = c0 + cc;
                let d = d0 + dd;
                var v: f32 = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                tile[j] = v;
            }
            workgroupBarrier();
            if (active) {
                let lim = min(256u, dp - d0);
                for (var k: u32 = 0u; k < 256u; k += 1u) {
                    if (k < lim) {
                        let pk = point_keys[base + d0 + k];
                        s0 += pk * tile[k];
                        s1 += pk * tile[256u + k];
                        s2 += pk * tile[512u + k];
                        s3 += pk * tile[768u + k];
                        s4 += pk * tile[1024u + k];
                        s5 += pk * tile[1280u + k];
                        s6 += pk * tile[1536u + k];
                        s7 += pk * tile[1792u + k];
                    }
                }
            }
        }
        if (active) {
            if (c0 < c && s0 > best_sim) { best_sim = s0; best = c0; }
            if (c0 + 1u < c && s1 > best_sim) { best_sim = s1; best = c0 + 1u; }
            if (c0 + 2u < c && s2 > best_sim) { best_sim = s2; best = c0 + 2u; }
            if (c0 + 3u < c && s3 > best_sim) { best_sim = s3; best = c0 + 3u; }
            if (c0 + 4u < c && s4 > best_sim) { best_sim = s4; best = c0 + 4u; }
            if (c0 + 5u < c && s5 > best_sim) { best_sim = s5; best = c0 + 5u; }
            if (c0 + 6u < c && s6 > best_sim) { best_sim = s6; best = c0 + 6u; }
            if (c0 + 7u < c && s7 > best_sim) { best_sim = s7; best = c0 + 7u; }
        }
    }

    if (active) {
        assignments[row_off + local] = best;
    }
}
