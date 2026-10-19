@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 512>;
var<workgroup> rs: array<f32, 512>;
var<workgroup> ri: array<u32, 512>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pb = wid.x * 64u;
    let tx = t & 7u;
    let ty = t >> 3u;

    var bs: array<f32, 8>;
    var bi: array<u32, 8>;
    for (var i = 0u; i < 8u; i += 1u) {
        bs[i] = -3.4e38;
        bi[i] = 0u;
    }

    for (var cb = 0u; cb < c; cb += 32u) {
        var acc: array<f32, 32>;
        for (var q = 0u; q < 32u; q += 1u) {
            acc[q] = 0.0;
        }
        for (var k0 = 0u; k0 < dp; k0 += 16u) {
            for (var e = 0u; e < 16u; e += 1u) {
                let idx = t + 64u * e;
                let row = idx >> 4u;
                let col = idx & 15u;
                var v = 0.0;
                if (pb + row < n && k0 + col < dp) {
                    v = point_keys[(pb + row) * dp + k0 + col];
                }
                As[col * 64u + row] = v;
            }
            for (var e = 0u; e < 8u; e += 1u) {
                let idx = t + 64u * e;
                let row = idx >> 4u;
                let col = idx & 15u;
                var v = 0.0;
                if (cb + row < c && k0 + col < dp) {
                    v = centroid_keys[(cb + row) * dp + k0 + col];
                }
                Bs[col * 32u + row] = v;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 16u; kk += 1u) {
                var a: array<f32, 8>;
                var b: array<f32, 4>;
                for (var i = 0u; i < 8u; i += 1u) {
                    a[i] = As[kk * 64u + ty + 8u * i];
                }
                for (var j = 0u; j < 4u; j += 1u) {
                    b[j] = Bs[kk * 32u + tx + 8u * j];
                }
                for (var i = 0u; i < 8u; i += 1u) {
                    for (var j = 0u; j < 4u; j += 1u) {
                        acc[i * 4u + j] = fma(a[i], b[j], acc[i * 4u + j]);
                    }
                }
            }
            workgroupBarrier();
        }
        for (var i = 0u; i < 8u; i += 1u) {
            for (var j = 0u; j < 4u; j += 1u) {
                let ci = cb + tx + 8u * j;
                let s = acc[i * 4u + j];
                if (ci < c && s > bs[i]) {
                    bs[i] = s;
                    bi[i] = ci;
                }
            }
        }
    }

    for (var i = 0u; i < 8u; i += 1u) {
        let r = (ty + 8u * i) * 8u + tx;
        rs[r] = bs[i];
        ri[r] = bi[i];
    }
    workgroupBarrier();

    let l = pb + t;
    if (l < n) {
        var best_s = rs[t * 8u];
        var best_i = ri[t * 8u];
        for (var x = 1u; x < 8u; x += 1u) {
            let s = rs[t * 8u + x];
            let ix = ri[t * 8u + x];
            if (s > best_s || (s == best_s && ix < best_i)) {
                best_s = s;
                best_i = ix;
            }
        }
        assignments[row_off + l] = best_i;
    }
}
