@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;
var<workgroup> rsim: array<f32, 256>;
var<workgroup> ridx: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pg = lid & 15u;
    let cg = lid >> 4u;
    let base = wid.x * 64u + pg * 4u;
    let nt = (c + 15u) / 16u;
    let nch = (dp + 63u) / 64u;

    var bs = vec4<f32>(-3.4e38);
    var bi = vec4<u32>(0u);

    for (var t = 0u; t < nt; t = t + 1u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var ch = 0u; ch < nch; ch = ch + 1u) {
            workgroupBarrier();
            for (var k = 0u; k < 4u; k = k + 1u) {
                let idx = k * 64u + lid;
                let cc = idx >> 4u;
                let d4 = idx & 15u;
                let ci = t * 16u + cc;
                let d = ch * 64u + d4 * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    for (var e = 0u; e < 4u; e = e + 1u) {
                        if (d + e < dp) {
                            v[e] = centroid_keys[ci * dp + d + e];
                        }
                    }
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var d4 = 0u; d4 < 16u; d4 = d4 + 1u) {
                let d = ch * 64u + d4 * 4u;
                var p0 = vec4<f32>(0.0);
                var p1 = vec4<f32>(0.0);
                var p2 = vec4<f32>(0.0);
                var p3 = vec4<f32>(0.0);
                if (d < dp) {
                    let full = d + 4u <= dp;
                    for (var e = 0u; e < 4u; e = e + 1u) {
                        if (d + e < dp) {
                            if (base < n) { p0[e] = point_keys[base * dp + d + e]; }
                            if (base + 1u < n) { p1[e] = point_keys[(base + 1u) * dp + d + e]; }
                            if (base + 2u < n) { p2[e] = point_keys[(base + 2u) * dp + d + e]; }
                            if (base + 3u < n) { p3[e] = point_keys[(base + 3u) * dp + d + e]; }
                        }
                    }
                    let tb = (cg * 4u) * 16u + d4;
                    let c0 = tile[tb];
                    let c1 = tile[tb + 16u];
                    let c2 = tile[tb + 32u];
                    let c3 = tile[tb + 48u];
                    a0 = a0 + vec4<f32>(dot(p0, c0), dot(p0, c1), dot(p0, c2), dot(p0, c3));
                    a1 = a1 + vec4<f32>(dot(p1, c0), dot(p1, c1), dot(p1, c2), dot(p1, c3));
                    a2 = a2 + vec4<f32>(dot(p2, c0), dot(p2, c1), dot(p2, c2), dot(p2, c3));
                    a3 = a3 + vec4<f32>(dot(p3, c0), dot(p3, c1), dot(p3, c2), dot(p3, c3));
                }
            }
        }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let ci = t * 16u + cg * 4u + j;
            if (ci < c) {
                if (a0[j] > bs.x) { bs.x = a0[j]; bi.x = ci; }
                if (a1[j] > bs.y) { bs.y = a1[j]; bi.y = ci; }
                if (a2[j] > bs.z) { bs.z = a2[j]; bi.z = ci; }
                if (a3[j] > bs.w) { bs.w = a3[j]; bi.w = ci; }
            }
        }
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        rsim[cg * 64u + pg * 4u + i] = bs[i];
        ridx[cg * 64u + pg * 4u + i] = bi[i];
    }
    workgroupBarrier();
    if (lid < 64u) {
        var s = rsim[lid];
        var ix = ridx[lid];
        for (var g = 1u; g < 4u; g = g + 1u) {
            let s2 = rsim[g * 64u + lid];
            let i2 = ridx[g * 64u + lid];
            if (s2 > s || (s2 == s && i2 < ix)) { s = s2; ix = i2; }
        }
        let l = wid.x * 64u + lid;
        if (l < n) {
            assignments[row_off + l] = ix;
        }
    }
}
