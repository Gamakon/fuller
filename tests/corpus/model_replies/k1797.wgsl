@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> sa: array<f32, 1088>;
var<workgroup> sb: array<f32, 272>;
var<workgroup> rs: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pbase = wid.x * 64u;
    let ty = lid >> 2u;
    let tx = lid & 3u;

    var bs = array<f32, 4>(-3.4e38, -3.4e38, -3.4e38, -3.4e38);
    var bi = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        var acc = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (var k0 = 0u; k0 < dp; k0 += 16u) {
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let e = lid + i * 64u;
                let row = e >> 4u;
                let k = e & 15u;
                var v = 0.0;
                if (pbase + row < n_band && k0 + k < dp) {
                    v = point_keys[(pbase + row) * dp + k0 + k];
                }
                sa[row * 17u + k] = v;
            }
            for (var i = 0u; i < 4u; i += 1u) {
                let e = lid + i * 64u;
                let row = e >> 4u;
                let k = e & 15u;
                var v = 0.0;
                if (c0 + row < c && k0 + k < dp) {
                    v = centroid_keys[(c0 + row) * dp + k0 + k];
                }
                sb[row * 17u + k] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 16u; k += 1u) {
                let a0 = sa[(ty * 4u) * 17u + k];
                let a1 = sa[(ty * 4u + 1u) * 17u + k];
                let a2 = sa[(ty * 4u + 2u) * 17u + k];
                let a3 = sa[(ty * 4u + 3u) * 17u + k];
                let b0 = sb[(tx * 4u) * 17u + k];
                let b1 = sb[(tx * 4u + 1u) * 17u + k];
                let b2 = sb[(tx * 4u + 2u) * 17u + k];
                let b3 = sb[(tx * 4u + 3u) * 17u + k];
                acc[0] += a0 * b0; acc[1] += a0 * b1; acc[2] += a0 * b2; acc[3] += a0 * b3;
                acc[4] += a1 * b0; acc[5] += a1 * b1; acc[6] += a1 * b2; acc[7] += a1 * b3;
                acc[8] += a2 * b0; acc[9] += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
                acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
            }
        }
        for (var j = 0u; j < 4u; j += 1u) {
            let ci = c0 + tx * 4u + j;
            if (ci < c) {
                for (var i = 0u; i < 4u; i += 1u) {
                    let s = acc[i * 4u + j];
                    if (s > bs[i]) { bs[i] = s; bi[i] = ci; }
                }
            }
        }
    }

    for (var i = 0u; i < 4u; i += 1u) {
        rs[(ty * 4u + i) * 4u + tx] = bs[i];
        ri[(ty * 4u + i) * 4u + tx] = bi[i];
    }
    workgroupBarrier();
    if (pbase + lid < n_band) {
        var s = rs[lid * 4u];
        var id = ri[lid * 4u];
        for (var t = 1u; t < 4u; t += 1u) {
            let s2 = rs[lid * 4u + t];
            let i2 = ri[lid * 4u + t];
            if (s2 > s || (s2 == s && i2 < id)) { s = s2; id = i2; }
        }
        assignments[row_off + pbase + lid] = id;
    }
}
