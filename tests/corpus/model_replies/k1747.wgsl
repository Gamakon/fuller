@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 256u;

    let l0 = base + lid;
    let l1 = l0 + 64u;
    let l2 = l0 + 128u;
    let l3 = l0 + 192u;
    let last = n - 1u;
    let r0 = min(l0, last) * dp;
    let r1 = min(l1, last) * dp;
    let r2 = min(l2, last) * dp;
    let r3 = min(l3, last) * dp;

    var best0: u32 = 0u;
    var best1: u32 = 0u;
    var best2: u32 = 0u;
    var best3: u32 = 0u;
    var bs0: f32 = -3.4e38;
    var bs1: f32 = -3.4e38;
    var bs2: f32 = -3.4e38;
    var bs3: f32 = -3.4e38;

    for (var ct0 = 0u; ct0 < c; ct0 += 16u) {
        var a0: array<f32, 16>;
        var a1: array<f32, 16>;
        var a2: array<f32, 16>;
        var a3: array<f32, 16>;
        for (var j = 0u; j < 16u; j += 1u) {
            a0[j] = 0.0;
            a1[j] = 0.0;
            a2[j] = 0.0;
            a3[j] = 0.0;
        }
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            for (var i = 0u; i < 1024u; i += 64u) {
                let idx = i + lid;
                let cc = idx >> 6u;
                let dd = d0 + (idx & 63u);
                let ci = ct0 + cc;
                var v: f32 = 0.0;
                if (ci < c && dd < dp) {
                    v = centroid_keys[ci * dp + dd];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            let dlim = min(64u, dp - d0);
            for (var d = 0u; d < dlim; d += 1u) {
                let p0 = point_keys[r0 + d0 + d];
                let p1 = point_keys[r1 + d0 + d];
                let p2 = point_keys[r2 + d0 + d];
                let p3 = point_keys[r3 + d0 + d];
                for (var j = 0u; j < 16u; j += 1u) {
                    let s = tile[j * 64u + d];
                    a0[j] = fma(p0, s, a0[j]);
                    a1[j] = fma(p1, s, a1[j]);
                    a2[j] = fma(p2, s, a2[j]);
                    a3[j] = fma(p3, s, a3[j]);
                }
            }
        }
        let jl = min(16u, c - ct0);
        for (var j = 0u; j < jl; j += 1u) {
            let ci = ct0 + j;
            if (a0[j] > bs0) { bs0 = a0[j]; best0 = ci; }
            if (a1[j] > bs1) { bs1 = a1[j]; best1 = ci; }
            if (a2[j] > bs2) { bs2 = a2[j]; best2 = ci; }
            if (a3[j] > bs3) { bs3 = a3[j]; best3 = ci; }
        }
    }

    if (l0 < n) { assignments[row_off + l0] = best0; }
    if (l1 < n) { assignments[row_off + l1] = best1; }
    if (l2 < n) { assignments[row_off + l2] = best2; }
    if (l3 < n) { assignments[row_off + l3] = best3; }
}
