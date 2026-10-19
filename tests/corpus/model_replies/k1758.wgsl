@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = local < n_band;
    let rowbase = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var ct = 0u; ct < c; ct += 8u) {
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
        for (var dt = 0u; dt < dp; dt += 128u) {
            for (var r = 0u; r < 2u; r += 1u) {
                let v = lid + r * 128u;
                let cc = v / 32u;
                let q = v % 32u;
                let gc = ct + cc;
                let gd = dt + q * 4u;
                var t = vec4<f32>(0.0);
                if (gc < c) {
                    let b = gc * dp + gd;
                    if (gd + 3u < dp) {
                        t = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
                    } else {
                        if (gd < dp) { t.x = centroid_keys[b]; }
                        if (gd + 1u < dp) { t.y = centroid_keys[b + 1u]; }
                        if (gd + 2u < dp) { t.z = centroid_keys[b + 2u]; }
                    }
                }
                tile[v] = t;
            }
            workgroupBarrier();
            if (valid) {
                for (var q = 0u; q < 32u; q += 1u) {
                    let gd = dt + q * 4u;
                    var pk = vec4<f32>(0.0);
                    if (gd + 3u < dp) {
                        let b = rowbase + gd;
                        pk = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                    } else {
                        if (gd < dp) { pk.x = point_keys[rowbase + gd]; }
                        if (gd + 1u < dp) { pk.y = point_keys[rowbase + gd + 1u]; }
                        if (gd + 2u < dp) { pk.z = point_keys[rowbase + gd + 2u]; }
                    }
                    a0 += dot(pk, tile[q]);
                    a1 += dot(pk, tile[32u + q]);
                    a2 += dot(pk, tile[64u + q]);
                    a3 += dot(pk, tile[96u + q]);
                    a4 += dot(pk, tile[128u + q]);
                    a5 += dot(pk, tile[160u + q]);
                    a6 += dot(pk, tile[192u + q]);
                    a7 += dot(pk, tile[224u + q]);
                }
            }
            workgroupBarrier();
        }
        if (ct < c && a0 > best_sim) { best_sim = a0; best = ct; }
        if (ct + 1u < c && a1 > best_sim) { best_sim = a1; best = ct + 1u; }
        if (ct + 2u < c && a2 > best_sim) { best_sim = a2; best = ct + 2u; }
        if (ct + 3u < c && a3 > best_sim) { best_sim = a3; best = ct + 3u; }
        if (ct + 4u < c && a4 > best_sim) { best_sim = a4; best = ct + 4u; }
        if (ct + 5u < c && a5 > best_sim) { best_sim = a5; best = ct + 5u; }
        if (ct + 6u < c && a6 > best_sim) { best_sim = a6; best = ct + 6u; }
        if (ct + 7u < c && a7 > best_sim) { best_sim = a7; best = ct + 7u; }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
