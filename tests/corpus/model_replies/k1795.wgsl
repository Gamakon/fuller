@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> ps: array<f32, 2048>;
var<workgroup> cs: array<f32, 1024>;
var<workgroup> ms: array<f32, 256>;
var<workgroup> mi: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 64u;
    let pg = t / 4u;
    let cq = t % 4u;

    var bsim: array<f32, 4>;
    var bidx: array<u32, 4>;
    for (var i = 0u; i < 4u; i += 1u) {
        bsim[i] = -3.4e38;
        bidx[i] = 0xFFFFFFFFu;
    }

    for (var c0 = 0u; c0 < c; c0 += 32u) {
        var acc: array<f32, 32>;
        for (var k = 0u; k < 32u; k += 1u) { acc[k] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 32u) {
            workgroupBarrier();
            for (var k = 0u; k < 32u; k += 1u) {
                let e = k * 64u + t;
                let pt = e / 32u;
                let dd = e % 32u;
                let l = base + pt;
                let d = d0 + dd;
                var v = 0.0;
                if (l < n && d < dp) { v = point_keys[l * dp + d]; }
                ps[e] = v;
            }
            for (var k = 0u; k < 16u; k += 1u) {
                let e = k * 64u + t;
                let cj = e / 32u;
                let dd = e % 32u;
                let ci = c0 + cj;
                let d = d0 + dd;
                var v = 0.0;
                if (ci < c && d < dp) { v = centroid_keys[ci * dp + d]; }
                cs[e] = v;
            }
            workgroupBarrier();
            for (var dd = 0u; dd < 32u; dd += 1u) {
                let p0 = ps[(pg * 4u + 0u) * 32u + dd];
                let p1 = ps[(pg * 4u + 1u) * 32u + dd];
                let p2 = ps[(pg * 4u + 2u) * 32u + dd];
                let p3 = ps[(pg * 4u + 3u) * 32u + dd];
                for (var j = 0u; j < 8u; j += 1u) {
                    let cv = cs[(cq * 8u + j) * 32u + dd];
                    acc[j * 4u + 0u] += p0 * cv;
                    acc[j * 4u + 1u] += p1 * cv;
                    acc[j * 4u + 2u] += p2 * cv;
                    acc[j * 4u + 3u] += p3 * cv;
                }
            }
        }

        for (var j = 0u; j < 8u; j += 1u) {
            let ci = c0 + cq * 8u + j;
            if (ci < c) {
                for (var i = 0u; i < 4u; i += 1u) {
                    let s = acc[j * 4u + i];
                    if (s > bsim[i]) {
                        bsim[i] = s;
                        bidx[i] = ci;
                    }
                }
            }
        }
    }

    for (var i = 0u; i < 4u; i += 1u) {
        let slot = (pg * 4u + i) * 4u + cq;
        ms[slot] = bsim[i];
        mi[slot] = bidx[i];
    }
    workgroupBarrier();

    let l = base + t;
    if (l < n) {
        var best = 0xFFFFFFFFu;
        var bs = -3.4e38;
        for (var q = 0u; q < 4u; q += 1u) {
            let s = ms[t * 4u + q];
            let ix = mi[t * 4u + q];
            if (s > bs || (s == bs && ix < best)) {
                bs = s;
                best = ix;
            }
        }
        if (best == 0xFFFFFFFFu) { best = 0u; }
        assignments[row_off + l] = best;
    }
}
