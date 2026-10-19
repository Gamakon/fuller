@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> A: array<f32, 2048>;
var<workgroup> B: array<f32, 1024>;
var<workgroup> mS: array<f32, 256>;
var<workgroup> mI: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 64u;
    let pg = lid & 15u;
    let cg = lid >> 4u;

    var bestS: array<f32, 4>;
    var bestI: array<u32, 4>;
    for (var i = 0u; i < 4u; i += 1u) {
        bestS[i] = -3.4e38;
        bestI[i] = 0u;
    }

    for (var c0 = 0u; c0 < c; c0 += 32u) {
        var acc: array<f32, 32>;
        for (var q = 0u; q < 32u; q += 1u) {
            acc[q] = 0.0;
        }
        for (var k0 = 0u; k0 < dp; k0 += 32u) {
            workgroupBarrier();
            for (var e = 0u; e < 32u; e += 1u) {
                let idx = e * 64u + lid;
                let pt = idx >> 5u;
                let k = idx & 31u;
                let gp = base + pt;
                var v = 0.0;
                if (gp < n && k0 + k < dp) {
                    v = point_keys[gp * dp + k0 + k];
                }
                A[k * 64u + pt] = v;
            }
            for (var e = 0u; e < 16u; e += 1u) {
                let idx = e * 64u + lid;
                let cc = idx >> 5u;
                let k = idx & 31u;
                let ci = c0 + cc;
                var v = 0.0;
                if (ci < c && k0 + k < dp) {
                    v = centroid_keys[ci * dp + k0 + k];
                }
                B[k * 32u + cc] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 32u; k += 1u) {
                var a: array<f32, 4>;
                for (var i = 0u; i < 4u; i += 1u) {
                    a[i] = A[k * 64u + pg + 16u * i];
                }
                for (var j = 0u; j < 8u; j += 1u) {
                    let b = B[k * 32u + cg * 8u + j];
                    for (var i = 0u; i < 4u; i += 1u) {
                        acc[i * 8u + j] = acc[i * 8u + j] + a[i] * b;
                    }
                }
            }
        }
        for (var j = 0u; j < 8u; j += 1u) {
            let ci = c0 + cg * 8u + j;
            if (ci < c) {
                for (var i = 0u; i < 4u; i += 1u) {
                    let s = acc[i * 8u + j];
                    if (s > bestS[i]) {
                        bestS[i] = s;
                        bestI[i] = ci;
                    }
                }
            }
        }
    }

    for (var i = 0u; i < 4u; i += 1u) {
        mS[cg * 64u + pg + 16u * i] = bestS[i];
        mI[cg * 64u + pg + 16u * i] = bestI[i];
    }
    workgroupBarrier();
    var bs = mS[lid];
    var bi = mI[lid];
    for (var g = 1u; g < 4u; g += 1u) {
        let s = mS[g * 64u + lid];
        let ix = mI[g * 64u + lid];
        if (s > bs || (s == bs && ix < bi)) {
            bs = s;
            bi = ix;
        }
    }
    if (base + lid < n) {
        assignments[row_off + base + lid] = bi;
    }
}
