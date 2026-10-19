@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let d4 = dp / 4u;
    let valid = gid.x < n_band;
    var prow = 0u;
    if (valid) { prow = gid.x * d4; }
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb = 0u; cb < c; cb += 8u) {
        var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        var s4 = 0.0; var s5 = 0.0; var s6 = 0.0; var s7 = 0.0;
        for (var d0 = 0u; d0 < d4; d0 += 32u) {
            workgroupBarrier();
            for (var i = 0u; i < 4u; i += 1u) {
                let idx = lid + i * 64u;
                let cc = idx / 32u;
                let q = idx % 32u;
                let ci = cb + cc;
                let dq = d0 + q;
                var v = vec4<f32>(0.0);
                if (ci < c && dq < d4) {
                    let b = (ci * d4 + dq) * 4u;
                    v = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var k = 0u; k < 32u; k += 1u) {
                    if (d0 + k < d4) {
                        let b = (prow + d0 + k) * 4u;
                        let p = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                        s0 += dot(p, tile[k]);
                        s1 += dot(p, tile[32u + k]);
                        s2 += dot(p, tile[64u + k]);
                        s3 += dot(p, tile[96u + k]);
                        s4 += dot(p, tile[128u + k]);
                        s5 += dot(p, tile[160u + k]);
                        s6 += dot(p, tile[192u + k]);
                        s7 += dot(p, tile[224u + k]);
                    }
                }
            }
        }
        if (cb + 0u < c && s0 > best_sim) { best_sim = s0; best = cb; }
        if (cb + 1u < c && s1 > best_sim) { best_sim = s1; best = cb + 1u; }
        if (cb + 2u < c && s2 > best_sim) { best_sim = s2; best = cb + 2u; }
        if (cb + 3u < c && s3 > best_sim) { best_sim = s3; best = cb + 3u; }
        if (cb + 4u < c && s4 > best_sim) { best_sim = s4; best = cb + 4u; }
        if (cb + 5u < c && s5 > best_sim) { best_sim = s5; best = cb + 5u; }
        if (cb + 6u < c && s6 > best_sim) { best_sim = s6; best = cb + 6u; }
        if (cb + 7u < c && s7 > best_sim) { best_sim = s7; best = cb + 7u; }
    }
    if (valid) {
        assignments[row_off + gid.x] = best;
    }
}
