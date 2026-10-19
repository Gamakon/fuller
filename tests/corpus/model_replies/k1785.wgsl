@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> sc: array<vec4<f32>, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let last = n_band - 1u;

    var rows: array<u32, 4>;
    for (var q = 0u; q < 4u; q += 1u) {
        rows[q] = min(wid.x * 256u + q * 64u + lid, last) * dp;
    }

    var bs: array<f32, 4>;
    var bi: array<u32, 4>;
    for (var q = 0u; q < 4u; q += 1u) {
        bs[q] = -3.4e38;
        bi[q] = 0u;
    }

    // loader mapping
    let lc = lid / 8u;
    let lj = lid % 8u;

    for (var cb = 0u; cb < c; cb += 8u) {
        var acc: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) { acc[i] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 32u) {
            // stage 8 centroids x 32 dims
            let gc = cb + lc;
            var v = vec4<f32>(0.0);
            let d = d0 + lj * 4u;
            if (gc < c) {
                let b = gc * dp + d;
                if (d + 0u < dp) { v.x = centroid_keys[b]; }
                if (d + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                if (d + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                if (d + 3u < dp) { v.w = centroid_keys[b + 3u]; }
            }
            sc[lid] = v;
            workgroupBarrier();

            for (var j = 0u; j < 8u; j += 1u) {
                let pd = d0 + j * 4u;
                var pv: array<vec4<f32>, 4>;
                for (var q = 0u; q < 4u; q += 1u) {
                    let b = rows[q] + pd;
                    pv[q] = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                }
                for (var k = 0u; k < 8u; k += 1u) {
                    let cv = sc[k * 8u + j];
                    for (var q = 0u; q < 4u; q += 1u) {
                        acc[q * 8u + k] += dot(pv[q], cv);
                    }
                }
            }
            workgroupBarrier();
        }

        for (var k = 0u; k < 8u; k += 1u) {
            let ci = cb + k;
            if (ci < c) {
                for (var q = 0u; q < 4u; q += 1u) {
                    let s = acc[q * 8u + k];
                    if (s > bs[q]) {
                        bs[q] = s;
                        bi[q] = ci;
                    }
                }
            }
        }
    }

    for (var q = 0u; q < 4u; q += 1u) {
        let l = wid.x * 256u + q * 64u + lid;
        if (l < n_band) {
            assignments[row_off + l] = bi[q];
        }
    }
}
