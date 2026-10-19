@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 512>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let active = local < n_band;
    let pbase = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0: u32 = 0u; c0 < c; c0 += 16u) {
        var s: array<f32, 16>;
        for (var i = 0u; i < 16u; i += 1u) { s[i] = 0.0; }
        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var k = 0u; k < 8u; k += 1u) {
                let idx = k * 64u + lid;
                let cc = idx / 32u;
                let dv = idx % 32u;
                let ci = c0 + cc;
                let d = d0 + dv * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c && d < dp) {
                    let b = ci * dp + d;
                    v = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u],
                                  centroid_keys[b + 2u], centroid_keys[b + 3u]);
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (active) {
                for (var j = 0u; j < 32u; j += 1u) {
                    let d = d0 + j * 4u;
                    if (d < dp) {
                        let b = pbase + d;
                        let pv = vec4<f32>(point_keys[b], point_keys[b + 1u],
                                           point_keys[b + 2u], point_keys[b + 3u]);
                        for (var i = 0u; i < 16u; i += 1u) {
                            s[i] += dot(pv, tile[i * 32u + j]);
                        }
                    }
                }
            }
        }
        for (var i = 0u; i < 16u; i += 1u) {
            if (c0 + i < c) {
                if (s[i] > best_sim) { best_sim = s[i]; best = c0 + i; }
            }
        }
    }
    if (active) {
        assignments[row_off + local] = best;
    }
}
