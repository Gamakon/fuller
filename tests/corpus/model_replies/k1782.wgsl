@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 2080>;
var<workgroup> bsim: array<f32, 256>;
var<workgroup> bidx: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pg = lid & 15u;
    let cs = lid >> 4u;
    let pbase = wid.x * 64u + pg * 4u;

    var base: array<u32, 4>;
    for (var i = 0u; i < 4u; i += 1u) {
        let l = pbase + i;
        base[i] = select(0u, l * dp, l < n_band);
    }
    var bs: array<f32, 4>;
    var bi: array<u32, 4>;
    for (var i = 0u; i < 4u; i += 1u) {
        bs[i] = -3.4e38;
        bi[i] = 0u;
    }

    for (var c0 = 0u; c0 < c; c0 += 32u) {
        var acc: array<f32, 32>;
        for (var j = 0u; j < 32u; j += 1u) { acc[j] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            let dn = min(64u, dp - d0);
            workgroupBarrier();
            for (var i = 0u; i < 32u; i += 1u) {
                let j = i * 64u + lid;
                let cc = j >> 6u;
                let dd = j & 63u;
                var v = 0.0;
                if (c0 + cc < c && dd < dn) {
                    v = centroid_keys[(c0 + cc) * dp + d0 + dd];
                }
                tile[cc * 65u + dd] = v;
            }
            workgroupBarrier();
            for (var d = 0u; d < 64u; d += 1u) {
                if (d < dn) {
                    let p0 = point_keys[base[0] + d0 + d];
                    let p1 = point_keys[base[1] + d0 + d];
                    let p2 = point_keys[base[2] + d0 + d];
                    let p3 = point_keys[base[3] + d0 + d];
                    for (var k = 0u; k < 8u; k += 1u) {
                        let cv = tile[(cs * 8u + k) * 65u + d];
                        acc[k * 4u + 0u] += p0 * cv;
                        acc[k * 4u + 1u] += p1 * cv;
                        acc[k * 4u + 2u] += p2 * cv;
                        acc[k * 4u + 3u] += p3 * cv;
                    }
                }
            }
        }

        for (var k = 0u; k < 8u; k += 1u) {
            let ci = c0 + cs * 8u + k;
            if (ci < c) {
                for (var i = 0u; i < 4u; i += 1u) {
                    let s = acc[k * 4u + i];
                    if (s > bs[i]) {
                        bs[i] = s;
                        bi[i] = ci;
                    }
                }
            }
        }
    }

    for (var i = 0u; i < 4u; i += 1u) {
        bsim[cs * 64u + pg * 4u + i] = bs[i];
        bidx[cs * 64u + pg * 4u + i] = bi[i];
    }
    workgroupBarrier();

    let l = wid.x * 64u + lid;
    if (l < n_band) {
        // lid = pg*4+i layout matches pg*4+i index of the point slot
        var s = bsim[lid];
        var b = bidx[lid];
        for (var q = 1u; q < 4u; q += 1u) {
            let s2 = bsim[q * 64u + lid];
            let b2 = bidx[q * 64u + lid];
            if (s2 > s || (s2 == s && b2 < b)) {
                s = s2;
                b = b2;
            }
        }
        assignments[row_off + l] = b;
    }
}
