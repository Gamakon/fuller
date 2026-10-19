@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;

    var base: array<u32, 4>;
    var best: array<u32, 4>;
    var bsim: array<f32, 4>;
    for (var j = 0u; j < 4u; j += 1u) {
        let l = wid.x * 256u + j * 64u + t;
        base[j] = min(l, n_band - 1u) * dp;
        best[j] = 0u;
        bsim[j] = -3.4e38;
    }

    for (var cb = 0u; cb < c; cb += 8u) {
        var acc: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) { acc[i] = 0.0; }

        for (var kb = 0u; kb < dp; kb += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + t;
                let cc = idx >> 7u;
                let k = idx & 127u;
                let ci = cb + cc;
                var v = 0.0;
                if (ci < c && kb + k < dp) {
                    v = centroid_keys[ci * dp + kb + k];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            let kn = min(128u, dp - kb);
            for (var k = 0u; k < 128u; k += 1u) {
                if (k < kn) {
                    let p0 = point_keys[base[0] + kb + k];
                    let p1 = point_keys[base[1] + kb + k];
                    let p2 = point_keys[base[2] + kb + k];
                    let p3 = point_keys[base[3] + kb + k];
                    for (var cc = 0u; cc < 8u; cc += 1u) {
                        let cv = tile[cc * 128u + k];
                        acc[cc] = acc[cc] + p0 * cv;
                        acc[8u + cc] = acc[8u + cc] + p1 * cv;
                        acc[16u + cc] = acc[16u + cc] + p2 * cv;
                        acc[24u + cc] = acc[24u + cc] + p3 * cv;
                    }
                }
            }
        }

        for (var cc = 0u; cc < 8u; cc += 1u) {
            let ci = cb + cc;
            if (ci < c) {
                for (var j = 0u; j < 4u; j += 1u) {
                    let s = acc[j * 8u + cc];
                    if (s > bsim[j]) { bsim[j] = s; best[j] = ci; }
                }
            }
        }
    }

    for (var j = 0u; j < 4u; j += 1u) {
        let l = wid.x * 256u + j * 64u + t;
        if (l < n_band) {
            assignments[row_off + l] = best[j];
        }
    }
}
