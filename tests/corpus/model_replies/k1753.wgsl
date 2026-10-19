@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> wsim: array<f32, 256>;
var<workgroup> wci: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pg = tid >> 2u;
    let sub = tid & 3u;
    let base = wid.x * 64u + pg * 4u;
    let last = n_band - 1u;
    let r0 = min(base, last) * dp;
    let r1 = min(base + 1u, last) * dp;
    let r2 = min(base + 2u, last) * dp;
    let r3 = min(base + 3u, last) * dp;
    let dp4 = dp & ~3u;

    var b0 = -3.4e38; var b1 = -3.4e38; var b2 = -3.4e38; var b3 = -3.4e38;
    var i0 = 0xFFFFFFFFu; var i1 = 0xFFFFFFFFu; var i2 = 0xFFFFFFFFu; var i3 = 0xFFFFFFFFu;

    for (var i = 0u; i < c; i += 4u) {
        let ci = i + sub;
        if (ci < c) {
            let cb = ci * dp;
            var a0 = vec4<f32>(0.0);
            var a1 = vec4<f32>(0.0);
            var a2 = vec4<f32>(0.0);
            var a3 = vec4<f32>(0.0);
            for (var p = 0u; p < dp4; p += 4u) {
                let cv = vec4<f32>(centroid_keys[cb + p], centroid_keys[cb + p + 1u],
                                   centroid_keys[cb + p + 2u], centroid_keys[cb + p + 3u]);
                let v0 = vec4<f32>(point_keys[r0 + p], point_keys[r0 + p + 1u],
                                   point_keys[r0 + p + 2u], point_keys[r0 + p + 3u]);
                let v1 = vec4<f32>(point_keys[r1 + p], point_keys[r1 + p + 1u],
                                   point_keys[r1 + p + 2u], point_keys[r1 + p + 3u]);
                let v2 = vec4<f32>(point_keys[r2 + p], point_keys[r2 + p + 1u],
                                   point_keys[r2 + p + 2u], point_keys[r2 + p + 3u]);
                let v3 = vec4<f32>(point_keys[r3 + p], point_keys[r3 + p + 1u],
                                   point_keys[r3 + p + 2u], point_keys[r3 + p + 3u]);
                a0 = fma(v0, cv, a0);
                a1 = fma(v1, cv, a1);
                a2 = fma(v2, cv, a2);
                a3 = fma(v3, cv, a3);
            }
            var s0 = a0.x + a0.y + a0.z + a0.w;
            var s1 = a1.x + a1.y + a1.z + a1.w;
            var s2 = a2.x + a2.y + a2.z + a2.w;
            var s3 = a3.x + a3.y + a3.z + a3.w;
            for (var p = dp4; p < dp; p += 1u) {
                let cv = centroid_keys[cb + p];
                s0 += point_keys[r0 + p] * cv;
                s1 += point_keys[r1 + p] * cv;
                s2 += point_keys[r2 + p] * cv;
                s3 += point_keys[r3 + p] * cv;
            }
            if (s0 > b0) { b0 = s0; i0 = ci; }
            if (s1 > b1) { b1 = s1; i1 = ci; }
            if (s2 > b2) { b2 = s2; i2 = ci; }
            if (s3 > b3) { b3 = s3; i3 = ci; }
        }
    }

    let o = tid * 4u;
    wsim[o] = b0; wsim[o + 1u] = b1; wsim[o + 2u] = b2; wsim[o + 3u] = b3;
    wci[o] = i0; wci[o + 1u] = i1; wci[o + 2u] = i2; wci[o + 3u] = i3;
    workgroupBarrier();

    if (sub == 0u) {
        for (var k = 0u; k < 4u; k += 1u) {
            var bs = -3.4e38;
            var bc = 0xFFFFFFFFu;
            for (var s = 0u; s < 4u; s += 1u) {
                let idx = ((pg * 4u + s) * 4u) + k;
                let sv = wsim[idx];
                let cv = wci[idx];
                if (sv > bs || (sv == bs && cv < bc)) { bs = sv; bc = cv; }
            }
            let l = base + k;
            if (l < n_band) {
                assignments[row_off + l] = select(bc, 0u, bc == 0xFFFFFFFFu);
            }
        }
    }
}
