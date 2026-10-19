@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> sc: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 256u;
    let l0 = base + lid;
    let l1 = base + 64u + lid;
    let l2 = base + 128u + lid;
    let l3 = base + 192u + lid;
    let r0 = min(l0, n - 1u) * dp;
    let r1 = min(l1, n - 1u) * dp;
    let r2 = min(l2, n - 1u) * dp;
    let r3 = min(l3, n - 1u) * dp;

    var bestv = vec4<f32>(-3.4e38);
    var besti = vec4<u32>(0u);

    for (var c0 = 0u; c0 < c; c0 += 8u) {
        var acc: array<vec4<f32>, 8>;
        for (var q = 0u; q < 8u; q += 1u) {
            acc[q] = vec4<f32>(0.0);
        }
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx / 128u;
                let kk = idx % 128u;
                let ci = c0 + cc;
                let d = d0 + kk;
                var v = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                sc[idx] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 128u; k += 1u) {
                let d = d0 + k;
                if (d < dp) {
                    let pv = vec4<f32>(point_keys[r0 + d], point_keys[r1 + d], point_keys[r2 + d], point_keys[r3 + d]);
                    for (var q = 0u; q < 8u; q += 1u) {
                        acc[q] = acc[q] + pv * sc[q * 128u + k];
                    }
                }
            }
        }
        for (var q = 0u; q < 8u; q += 1u) {
            let ci = c0 + q;
            if (ci < c) {
                let s = acc[q];
                if (s.x > bestv.x) { bestv.x = s.x; besti.x = ci; }
                if (s.y > bestv.y) { bestv.y = s.y; besti.y = ci; }
                if (s.z > bestv.z) { bestv.z = s.z; besti.z = ci; }
                if (s.w > bestv.w) { bestv.w = s.w; besti.w = ci; }
            }
        }
    }

    if (l0 < n) { assignments[row_off + l0] = besti.x; }
    if (l1 < n) { assignments[row_off + l1] = besti.y; }
    if (l2 < n) { assignments[row_off + l2] = besti.z; }
    if (l3 < n) { assignments[row_off + l3] = besti.w; }
}
