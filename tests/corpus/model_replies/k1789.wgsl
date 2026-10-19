@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 512>;
var<workgroup> rsim: array<f32, 256>;
var<workgroup> ridx: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pl = lid % 16u;
    let cl = lid / 16u;
    let pbase = wid.x * 64u + pl * 4u;

    var base: array<u32, 4>;
    for (var k = 0u; k < 4u; k += 1u) {
        let l = min(pbase + k, max(n_band, 1u) - 1u);
        base[k] = l * dp;
    }
    var best_sim: array<f32, 4>;
    var best: array<u32, 4>;
    for (var k = 0u; k < 4u; k += 1u) {
        best_sim[k] = -3.4e38;
        best[k] = 0u;
    }
    var acc: array<f32, 16>;

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        for (var k = 0u; k < 16u; k += 1u) { acc[k] = 0.0; }
        for (var d0 = 0u; d0 < dp; d0 += 32u) {
            workgroupBarrier();
            for (var i = 0u; i < 8u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx / 32u;
                let dd = idx % 32u;
                var v = 0.0;
                if (c0 + cc < c && d0 + dd < dp) {
                    v = centroid_keys[(c0 + cc) * dp + d0 + dd];
                }
                tile[dd * 16u + cc] = v;
            }
            workgroupBarrier();
            for (var j = 0u; j < 32u; j += 1u) {
                let d = d0 + j;
                if (d < dp) {
                    let p0 = point_keys[base[0] + d];
                    let p1 = point_keys[base[1] + d];
                    let p2 = point_keys[base[2] + d];
                    let p3 = point_keys[base[3] + d];
                    for (var cc = 0u; cc < 4u; cc += 1u) {
                        let w = tile[j * 16u + cl * 4u + cc];
                        acc[cc] += p0 * w;
                        acc[4u + cc] += p1 * w;
                        acc[8u + cc] += p2 * w;
                        acc[12u + cc] += p3 * w;
                    }
                }
            }
        }
        for (var cc = 0u; cc < 4u; cc += 1u) {
            let ci = c0 + cl * 4u + cc;
            if (ci < c) {
                for (var k = 0u; k < 4u; k += 1u) {
                    let s = acc[k * 4u + cc];
                    if (s > best_sim[k]) {
                        best_sim[k] = s;
                        best[k] = ci;
                    }
                }
            }
        }
    }

    for (var k = 0u; k < 4u; k += 1u) {
        rsim[lid * 4u + k] = best_sim[k];
        ridx[lid * 4u + k] = best[k];
    }
    workgroupBarrier();
    if (lid < 16u) {
        for (var k = 0u; k < 4u; k += 1u) {
            var bs = rsim[lid * 4u + k];
            var bi = ridx[lid * 4u + k];
            for (var q = 1u; q < 4u; q += 1u) {
                let o = (q * 16u + lid) * 4u + k;
                let s = rsim[o];
                let i2 = ridx[o];
                if (s > bs || (s == bs && i2 < bi)) {
                    bs = s;
                    bi = i2;
                }
            }
            let l = pbase + k;
            if (l < n_band) {
                assignments[row_off + l] = bi;
            }
        }
    }
}
