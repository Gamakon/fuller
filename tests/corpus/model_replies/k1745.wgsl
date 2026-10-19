@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 512>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 256u;
    let r0 = base + lid;
    let r1 = r0 + 64u;
    let r2 = r0 + 128u;
    let r3 = r0 + 192u;
    let v0 = r0 < n_band;
    let v1 = r1 < n_band;
    let v2 = r2 < n_band;
    let v3 = r3 < n_band;

    var bs = vec4<f32>(-3.4e38);
    var bi = vec4<u32>(0u);

    for (var ct = 0u; ct < c; ct += 16u) {
        var acc: array<vec4<f32>, 16>;
        for (var q = 0u; q < 16u; q += 1u) {
            acc[q] = vec4<f32>(0.0);
        }
        for (var dt = 0u; dt < dp; dt += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 8u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx >> 5u;
                let d4 = idx & 31u;
                let ci = ct + cc;
                let dd = dt + d4 * 4u;
                var t = vec4<f32>(0.0);
                if (ci < c) {
                    let o = ci * dp + dd;
                    if (dd + 0u < dp) { t.x = centroid_keys[o]; }
                    if (dd + 1u < dp) { t.y = centroid_keys[o + 1u]; }
                    if (dd + 2u < dp) { t.z = centroid_keys[o + 2u]; }
                    if (dd + 3u < dp) { t.w = centroid_keys[o + 3u]; }
                }
                tile[idx] = t;
            }
            workgroupBarrier();
            for (var d4 = 0u; d4 < 32u; d4 += 1u) {
                let dd = dt + d4 * 4u;
                var p0 = vec4<f32>(0.0);
                var p1 = vec4<f32>(0.0);
                var p2 = vec4<f32>(0.0);
                var p3 = vec4<f32>(0.0);
                if (dd + 3u < dp) {
                    if (v0) {
                        let o = r0 * dp + dd;
                        p0 = vec4<f32>(point_keys[o], point_keys[o + 1u], point_keys[o + 2u], point_keys[o + 3u]);
                    }
                    if (v1) {
                        let o = r1 * dp + dd;
                        p1 = vec4<f32>(point_keys[o], point_keys[o + 1u], point_keys[o + 2u], point_keys[o + 3u]);
                    }
                    if (v2) {
                        let o = r2 * dp + dd;
                        p2 = vec4<f32>(point_keys[o], point_keys[o + 1u], point_keys[o + 2u], point_keys[o + 3u]);
                    }
                    if (v3) {
                        let o = r3 * dp + dd;
                        p3 = vec4<f32>(point_keys[o], point_keys[o + 1u], point_keys[o + 2u], point_keys[o + 3u]);
                    }
                } else if (dd < dp) {
                    for (var j = 0u; j < 4u; j += 1u) {
                        if (dd + j < dp) {
                            if (v0) { p0[j] = point_keys[r0 * dp + dd + j]; }
                            if (v1) { p1[j] = point_keys[r1 * dp + dd + j]; }
                            if (v2) { p2[j] = point_keys[r2 * dp + dd + j]; }
                            if (v3) { p3[j] = point_keys[r3 * dp + dd + j]; }
                        }
                    }
                }
                for (var q = 0u; q < 16u; q += 1u) {
                    let cv = tile[q * 32u + d4];
                    acc[q] += vec4<f32>(dot(p0, cv), dot(p1, cv), dot(p2, cv), dot(p3, cv));
                }
            }
        }
        for (var q = 0u; q < 16u; q += 1u) {
            let ci = ct + q;
            if (ci < c) {
                let sim = acc[q];
                let gt = sim > bs;
                bs = select(bs, sim, gt);
                bi = select(bi, vec4<u32>(ci), gt);
            }
        }
    }

    if (v0) { assignments[row_off + r0] = bi.x; }
    if (v1) { assignments[row_off + r1] = bi.y; }
    if (v2) { assignments[row_off + r2] = bi.z; }
    if (v3) { assignments[row_off + r3] = bi.w; }
}
