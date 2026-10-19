@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> ssim: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let t = lid3.x;
    let pg = t % 16u;
    let cs = t / 16u;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 64u + pg * 4u;

    let l0 = min(base, n_band - 1u) * dp;
    let l1 = min(base + 1u, n_band - 1u) * dp;
    let l2 = min(base + 2u, n_band - 1u) * dp;
    let l3 = min(base + 3u, n_band - 1u) * dp;

    var b0: f32 = -3.4e38; var i0: u32 = 0u;
    var b1: f32 = -3.4e38; var i1: u32 = 0u;
    var b2: f32 = -3.4e38; var i2: u32 = 0u;
    var b3: f32 = -3.4e38; var i3: u32 = 0u;

    let ngroups = (c + 3u) / 4u;
    for (var k = 0u; k < ngroups; k += 4u) {
        let g = k + cs;
        if (g < ngroups) {
            let cb = g * 4u;
            let ca = min(cb, c - 1u) * dp;
            let cb1 = min(cb + 1u, c - 1u) * dp;
            let cc = min(cb + 2u, c - 1u) * dp;
            let cd = min(cb + 3u, c - 1u) * dp;
            var s00 = 0.0; var s01 = 0.0; var s02 = 0.0; var s03 = 0.0;
            var s10 = 0.0; var s11 = 0.0; var s12 = 0.0; var s13 = 0.0;
            var s20 = 0.0; var s21 = 0.0; var s22 = 0.0; var s23 = 0.0;
            var s30 = 0.0; var s31 = 0.0; var s32 = 0.0; var s33 = 0.0;
            for (var p = 0u; p < dp; p += 1u) {
                let a0 = point_keys[l0 + p];
                let a1 = point_keys[l1 + p];
                let a2 = point_keys[l2 + p];
                let a3 = point_keys[l3 + p];
                let x0 = centroid_keys[ca + p];
                let x1 = centroid_keys[cb1 + p];
                let x2 = centroid_keys[cc + p];
                let x3 = centroid_keys[cd + p];
                s00 += a0 * x0; s01 += a0 * x1; s02 += a0 * x2; s03 += a0 * x3;
                s10 += a1 * x0; s11 += a1 * x1; s12 += a1 * x2; s13 += a1 * x3;
                s20 += a2 * x0; s21 += a2 * x1; s22 += a2 * x2; s23 += a2 * x3;
                s30 += a3 * x0; s31 += a3 * x1; s32 += a3 * x2; s33 += a3 * x3;
            }
            if (cb < c) {
                if (s00 > b0) { b0 = s00; i0 = cb; }
                if (s10 > b1) { b1 = s10; i1 = cb; }
                if (s20 > b2) { b2 = s20; i2 = cb; }
                if (s30 > b3) { b3 = s30; i3 = cb; }
            }
            if (cb + 1u < c) {
                if (s01 > b0) { b0 = s01; i0 = cb + 1u; }
                if (s11 > b1) { b1 = s11; i1 = cb + 1u; }
                if (s21 > b2) { b2 = s21; i2 = cb + 1u; }
                if (s31 > b3) { b3 = s31; i3 = cb + 1u; }
            }
            if (cb + 2u < c) {
                if (s02 > b0) { b0 = s02; i0 = cb + 2u; }
                if (s12 > b1) { b1 = s12; i1 = cb + 2u; }
                if (s22 > b2) { b2 = s22; i2 = cb + 2u; }
                if (s32 > b3) { b3 = s32; i3 = cb + 2u; }
            }
            if (cb + 3u < c) {
                if (s03 > b0) { b0 = s03; i0 = cb + 3u; }
                if (s13 > b1) { b1 = s13; i1 = cb + 3u; }
                if (s23 > b2) { b2 = s23; i2 = cb + 3u; }
                if (s33 > b3) { b3 = s33; i3 = cb + 3u; }
            }
        }
    }

    let o = cs * 64u + pg * 4u;
    ssim[o] = b0; sidx[o] = i0;
    ssim[o + 1u] = b1; sidx[o + 1u] = i1;
    ssim[o + 2u] = b2; sidx[o + 2u] = i2;
    ssim[o + 3u] = b3; sidx[o + 3u] = i3;
    workgroupBarrier();

    let local = wid.x * 64u + t;
    var bs: f32 = ssim[t];
    var bi: u32 = sidx[t];
    for (var q = 1u; q < 4u; q += 1u) {
        let s = ssim[q * 64u + t];
        let ix = sidx[q * 64u + t];
        if (s > bs || (s == bs && ix < bi)) { bs = s; bi = ix; }
    }
    if (local < n_band) {
        assignments[row_off + local] = bi;
    }
}
