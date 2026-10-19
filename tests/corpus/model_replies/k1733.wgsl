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
    let valid = gid.x < n_band;
    let dp4 = (dp + 3u) / 4u;
    let pbase = gid.x * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cg = 0u; cg < c; cg += 8u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        var a4 = vec4<f32>(0.0);
        var a5 = vec4<f32>(0.0);
        var a6 = vec4<f32>(0.0);
        var a7 = vec4<f32>(0.0);
        for (var v0 = 0u; v0 < dp4; v0 += 32u) {
            for (var i = 0u; i < 4u; i += 1u) {
                let e = i * 64u + lid;
                let cc = e >> 5u;
                let v = e & 31u;
                let ci = cg + cc;
                let vv = v0 + v;
                var t = vec4<f32>(0.0);
                if (ci < c && vv < dp4) {
                    let b = ci * dp + vv * 4u;
                    let d0 = vv * 4u;
                    t.x = centroid_keys[b];
                    if (d0 + 1u < dp) { t.y = centroid_keys[b + 1u]; }
                    if (d0 + 2u < dp) { t.z = centroid_keys[b + 2u]; }
                    if (d0 + 3u < dp) { t.w = centroid_keys[b + 3u]; }
                }
                tile[e] = t;
            }
            workgroupBarrier();
            for (var v = 0u; v < 32u; v += 1u) {
                let vv = v0 + v;
                if (vv < dp4) {
                    var p = vec4<f32>(0.0);
                    if (valid) {
                        let d0 = vv * 4u;
                        let b = pbase + d0;
                        p.x = point_keys[b];
                        if (d0 + 1u < dp) { p.y = point_keys[b + 1u]; }
                        if (d0 + 2u < dp) { p.z = point_keys[b + 2u]; }
                        if (d0 + 3u < dp) { p.w = point_keys[b + 3u]; }
                    }
                    a0 += p * tile[v];
                    a1 += p * tile[32u + v];
                    a2 += p * tile[64u + v];
                    a3 += p * tile[96u + v];
                    a4 += p * tile[128u + v];
                    a5 += p * tile[160u + v];
                    a6 += p * tile[192u + v];
                    a7 += p * tile[224u + v];
                }
            }
            workgroupBarrier();
        }
        var s: array<f32, 8>;
        s[0] = (a0.x + a0.y) + (a0.z + a0.w);
        s[1] = (a1.x + a1.y) + (a1.z + a1.w);
        s[2] = (a2.x + a2.y) + (a2.z + a2.w);
        s[3] = (a3.x + a3.y) + (a3.z + a3.w);
        s[4] = (a4.x + a4.y) + (a4.z + a4.w);
        s[5] = (a5.x + a5.y) + (a5.z + a5.w);
        s[6] = (a6.x + a6.y) + (a6.z + a6.w);
        s[7] = (a7.x + a7.y) + (a7.z + a7.w);
        for (var k = 0u; k < 8u; k += 1u) {
            if (cg + k < c) {
                if (s[k] > best_sim) { best_sim = s[k]; best = cg + k; }
            }
        }
    }
    if (valid) {
        assignments[row_off + gid.x] = best;
    }
}
