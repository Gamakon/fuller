@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

const CT: u32 = 16u;
const DK4: u32 = 32u;
var<workgroup> tile: array<vec4<f32>, 512>;

fn load4(isp: bool, base: u32, d: u32, dp: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    if (d + 3u < dp) {
        if (isp) {
            v = vec4<f32>(point_keys[base + d], point_keys[base + d + 1u], point_keys[base + d + 2u], point_keys[base + d + 3u]);
        } else {
            v = vec4<f32>(centroid_keys[base + d], centroid_keys[base + d + 1u], centroid_keys[base + d + 2u], centroid_keys[base + d + 3u]);
        }
    } else {
        for (var q = 0u; q < 4u; q += 1u) {
            if (d + q < dp) {
                if (isp) { v[q] = point_keys[base + d + q]; } else { v[q] = centroid_keys[base + d + q]; }
            }
        }
    }
    return v;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let l0 = wid.x * 128u + lid;
    let l1 = l0 + 64u;
    let r0 = min(l0, n - 1u);
    let r1 = min(l1, n - 1u);
    let pb0 = r0 * dp;
    let pb1 = r1 * dp;
    var best0: u32 = 0u;
    var best1: u32 = 0u;
    var bs0: f32 = -3.4e38;
    var bs1: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += CT) {
        var a0: array<f32, 16>;
        var a1: array<f32, 16>;
        for (var j = 0u; j < CT; j += 1u) { a0[j] = 0.0; a1[j] = 0.0; }
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            for (var i = 0u; i < 8u; i += 1u) {
                let idx = i * 64u + lid;
                let cj = idx / DK4;
                let k = idx % DK4;
                let ci = c0 + cj;
                var v = vec4<f32>(0.0);
                let d = d0 + 4u * k;
                if (ci < c && d < dp) {
                    v = load4(false, ci * dp, d, dp);
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < DK4; k += 1u) {
                let d = d0 + 4u * k;
                if (d < dp) {
                    let p0 = load4(true, pb0, d, dp);
                    let p1 = load4(true, pb1, d, dp);
                    for (var j = 0u; j < CT; j += 1u) {
                        let t = tile[j * DK4 + k];
                        a0[j] += dot(p0, t);
                        a1[j] += dot(p1, t);
                    }
                }
            }
            workgroupBarrier();
        }
        for (var j = 0u; j < CT; j += 1u) {
            let ci = c0 + j;
            if (ci < c) {
                if (a0[j] > bs0) { bs0 = a0[j]; best0 = ci; }
                if (a1[j] > bs1) { bs1 = a1[j]; best1 = ci; }
            }
        }
    }
    if (l0 < n) { assignments[row_off + l0] = best0; }
    if (l1 < n) { assignments[row_off + l1] = best1; }
}
