@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> ptile: array<f32, 2112>;
var<workgroup> ctile: array<f32, 528>;
var<workgroup> bs: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 64u;
    let pg = t & 15u;
    let cs = t >> 4u;

    var best_s: array<f32, 4>;
    var best_i: array<u32, 4>;
    for (var i = 0u; i < 4u; i += 1u) {
        best_s[i] = -3.4e38;
        best_i[i] = 0u;
    }

    let nblocks = (c + 15u) / 16u;
    let nchunks = (dp + 31u) / 32u;

    for (var cb = 0u; cb < nblocks; cb += 1u) {
        var acc: array<f32, 16>;
        for (var k = 0u; k < 16u; k += 1u) { acc[k] = 0.0; }
        let cbase = cb * 16u;

        for (var dc = 0u; dc < nchunks; dc += 1u) {
            let d0 = dc * 32u;
            workgroupBarrier();
            for (var k = 0u; k < 32u; k += 1u) {
                let idx = k * 64u + t;
                let row = idx >> 5u;
                let col = idx & 31u;
                var v = 0.0;
                if (base + row < n_band && d0 + col < dp) {
                    v = point_keys[(base + row) * dp + d0 + col];
                }
                ptile[row * 33u + col] = v;
            }
            for (var k = 0u; k < 8u; k += 1u) {
                let idx = k * 64u + t;
                let row = idx >> 5u;
                let col = idx & 31u;
                var v = 0.0;
                if (cbase + row < c && d0 + col < dp) {
                    v = centroid_keys[(cbase + row) * dp + d0 + col];
                }
                ctile[row * 33u + col] = v;
            }
            workgroupBarrier();
            for (var d = 0u; d < 32u; d += 1u) {
                var pv: array<f32, 4>;
                for (var i = 0u; i < 4u; i += 1u) {
                    pv[i] = ptile[(pg * 4u + i) * 33u + d];
                }
                for (var j = 0u; j < 4u; j += 1u) {
                    let cv = ctile[(cs + 4u * j) * 33u + d];
                    for (var i = 0u; i < 4u; i += 1u) {
                        acc[j * 4u + i] = acc[j * 4u + i] + pv[i] * cv;
                    }
                }
            }
        }

        for (var j = 0u; j < 4u; j += 1u) {
            let ci = cbase + cs + 4u * j;
            if (ci < c) {
                for (var i = 0u; i < 4u; i += 1u) {
                    let s = acc[j * 4u + i];
                    if (s > best_s[i]) {
                        best_s[i] = s;
                        best_i[i] = ci;
                    }
                }
            }
        }
    }

    for (var i = 0u; i < 4u; i += 1u) {
        let slot = cs * 64u + pg * 4u + i;
        bs[slot] = best_s[i];
        bi[slot] = best_i[i];
    }
    workgroupBarrier();
    if (t < 64u && base + t < n_band) {
        var s = bs[t];
        var ix = bi[t];
        for (var q = 1u; q < 4u; q += 1u) {
            let s2 = bs[q * 64u + t];
            let i2 = bi[q * 64u + t];
            if (s2 > s || (s2 == s && i2 < ix)) {
                s = s2;
                ix = i2;
            }
        }
        assignments[row_off + base + t] = ix;
    }
}
