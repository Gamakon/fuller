@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lidv: vec3<u32>) {
    let lid = lidv.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = local < n_band;
    let prow = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    for (var cb: u32 = 0u; cb < c; cb += 8u) {
        var acc: array<f32, 8>;
        for (var k = 0u; k < 8u; k++) { acc[k] = 0.0; }
        for (var d0: u32 = 0u; d0 < dp; d0 += 32u) {
            workgroupBarrier();
            {
                let k = lid >> 3u;
                let q = (lid & 7u) * 4u;
                let ci = cb + k;
                var v = vec4<f32>(0.0);
                for (var j = 0u; j < 4u; j++) {
                    let d = d0 + q + j;
                    if (ci < c && d < dp) {
                        v[j] = centroid_keys[ci * dp + d];
                    }
                }
                tile[lid] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var q = 0u; q < 8u; q++) {
                    var pv = vec4<f32>(0.0);
                    for (var j = 0u; j < 4u; j++) {
                        let d = d0 + q * 4u + j;
                        if (d < dp) {
                            pv[j] = point_keys[prow + d];
                        }
                    }
                    for (var k = 0u; k < 8u; k++) {
                        acc[k] += dot(pv, tile[k * 8u + q]);
                    }
                }
            }
        }
        for (var k = 0u; k < 8u; k++) {
            if (cb + k < c) {
                if (acc[k] > best_sim) { best_sim = acc[k]; best = cb + k; }
            }
        }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
