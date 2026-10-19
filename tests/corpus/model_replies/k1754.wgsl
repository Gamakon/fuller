var<workgroup> tile: array<f32, 1024>;

@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = local < n_band;
    let rowbase = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 8u) {
        var acc: array<f32, 8>;
        for (var k = 0u; k < 8u; k += 1u) { acc[k] = 0.0; }
        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + lid;
                let cj = idx >> 7u;
                let d = idx & 127u;
                let gc = cb + cj;
                let gd = d0 + d;
                var v: f32 = 0.0;
                if (gc < c && gd < dp) { v = centroid_keys[gc * dp + gd]; }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var d = 0u; d < 128u; d += 1u) {
                    let gd = d0 + d;
                    if (gd < dp) {
                        let x = point_keys[rowbase + gd];
                        for (var k = 0u; k < 8u; k += 1u) {
                            acc[k] = acc[k] + x * tile[k * 128u + d];
                        }
                    }
                }
            }
        }
        for (var k = 0u; k < 8u; k += 1u) {
            if (cb + k < c && acc[k] > best_sim) { best_sim = acc[k]; best = cb + k; }
        }
    }
    if (valid) { assignments[row_off + local] = best; }
}
