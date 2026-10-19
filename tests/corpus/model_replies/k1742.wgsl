@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let l0 = wid.x * 128u + lid;
    let l1 = l0 + 64u;
    let v0 = l0 < n_band;
    let v1 = l1 < n_band;
    let r0 = min(l0, n_band - 1u) * dp;
    let r1 = min(l1, n_band - 1u) * dp;

    var best0: u32 = 0u;
    var best1: u32 = 0u;
    var bs0: f32 = -3.4e38;
    var bs1: f32 = -3.4e38;

    for (var cb = 0u; cb < c; cb += 4u) {
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
        for (var dd = 0u; dd < dp; dd += 256u) {
            let n = min(256u, dp - dd);
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + lid;
                let k = idx >> 8u;
                let j = idx & 255u;
                let ci = cb + k;
                var val = 0.0;
                if (ci < c && j < n) {
                    val = centroid_keys[ci * dp + dd + j];
                }
                tile[idx] = val;
            }
            workgroupBarrier();
            for (var j = 0u; j < 256u; j += 1u) {
                if (j < n) {
                    let p0 = point_keys[r0 + dd + j];
                    let p1 = point_keys[r1 + dd + j];
                    let t0 = tile[j];
                    let t1 = tile[256u + j];
                    let t2 = tile[512u + j];
                    let t3 = tile[768u + j];
                    a0 += p0 * t0; a1 += p0 * t1; a2 += p0 * t2; a3 += p0 * t3;
                    b0 += p1 * t0; b1 += p1 * t1; b2 += p1 * t2; b3 += p1 * t3;
                }
            }
        }
        if (a0 > bs0) { bs0 = a0; best0 = cb; }
        if (cb + 1u < c && a1 > bs0) { bs0 = a1; best0 = cb + 1u; }
        if (cb + 2u < c && a2 > bs0) { bs0 = a2; best0 = cb + 2u; }
        if (cb + 3u < c && a3 > bs0) { bs0 = a3; best0 = cb + 3u; }
        if (b0 > bs1) { bs1 = b0; best1 = cb; }
        if (cb + 1u < c && b1 > bs1) { bs1 = b1; best1 = cb + 1u; }
        if (cb + 2u < c && b2 > bs1) { bs1 = b2; best1 = cb + 2u; }
        if (cb + 3u < c && b3 > bs1) { bs1 = b3; best1 = cb + 3u; }
    }
    if (v0) { assignments[row_off + l0] = best0; }
    if (v1) { assignments[row_off + l1] = best1; }
}
