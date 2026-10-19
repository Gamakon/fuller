@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 64>;

fn ld_p(l: u32, d: u32, n: u32, dp: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    if (l < n) {
        let b = l * dp + d;
        if (d + 3u < dp) {
            v = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
        } else {
            if (d < dp) { v.x = point_keys[b]; }
            if (d + 1u < dp) { v.y = point_keys[b + 1u]; }
            if (d + 2u < dp) { v.z = point_keys[b + 2u]; }
        }
    }
    return v;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let t = lid.x;
    let l0 = wid.x * 128u + t;
    let l1 = l0 + 64u;

    var best0: u32 = 0u;
    var best1: u32 = 0u;
    var bs0: f32 = -3.4e38;
    var bs1: f32 = -3.4e38;

    let cc = t >> 4u;
    let dq = t & 15u;

    for (var c0 = 0u; c0 < c; c0 += 4u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        var b0 = vec4<f32>(0.0);
        var b1 = vec4<f32>(0.0);
        var b2 = vec4<f32>(0.0);
        var b3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            {
                let ci = c0 + cc;
                let d = d0 + dq * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let b = ci * dp + d;
                    if (d + 3u < dp) {
                        v = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
                    } else {
                        if (d < dp) { v.x = centroid_keys[b]; }
                        if (d + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                        if (d + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                    }
                }
                tile[t] = v;
            }
            workgroupBarrier();
            for (var q = 0u; q < 16u; q += 1u) {
                let d = d0 + q * 4u;
                let p0 = ld_p(l0, d, n, dp);
                let p1 = ld_p(l1, d, n, dp);
                let k0 = tile[q];
                let k1 = tile[16u + q];
                let k2 = tile[32u + q];
                let k3 = tile[48u + q];
                a0 += p0 * k0; a1 += p0 * k1; a2 += p0 * k2; a3 += p0 * k3;
                b0 += p1 * k0; b1 += p1 * k1; b2 += p1 * k2; b3 += p1 * k3;
            }
        }
        let sa = vec4<f32>(dot(a0, vec4<f32>(1.0)), dot(a1, vec4<f32>(1.0)), dot(a2, vec4<f32>(1.0)), dot(a3, vec4<f32>(1.0)));
        let sb = vec4<f32>(dot(b0, vec4<f32>(1.0)), dot(b1, vec4<f32>(1.0)), dot(b2, vec4<f32>(1.0)), dot(b3, vec4<f32>(1.0)));
        for (var j = 0u; j < 4u; j += 1u) {
            if (c0 + j < c) {
                if (sa[j] > bs0) { bs0 = sa[j]; best0 = c0 + j; }
                if (sb[j] > bs1) { bs1 = sb[j]; best1 = c0 + j; }
            }
        }
    }
    if (l0 < n) { assignments[row_off + l0] = best0; }
    if (l1 < n) { assignments[row_off + l1] = best1; }
}
