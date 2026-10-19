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
    let lp = select(0u, local, valid);
    let pbase = lp * dp;
    let dlast = dp - 1u;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        var acc: array<f32, 16>;
        for (var j = 0u; j < 16u; j += 1u) { acc[j] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            for (var k = 0u; k < 2u; k += 1u) {
                let idx = k * 128u + lid;
                let j = idx / 16u;
                let p4 = idx % 16u;
                let ci = c0 + j;
                let dd = d0 + p4 * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let b = ci * dp + dd;
                    if (dd < dp) { v.x = centroid_keys[b]; }
                    if (dd + 1u < dp) { v.y = centroid_keys[b + 1u]; }
                    if (dd + 2u < dp) { v.z = centroid_keys[b + 2u]; }
                    if (dd + 3u < dp) { v.w = centroid_keys[b + 3u]; }
                }
                tile[idx] = v;
            }
            workgroupBarrier();

            for (var p4 = 0u; p4 < 16u; p4 += 1u) {
                let dd = d0 + p4 * 4u;
                let x = vec4<f32>(
                    point_keys[pbase + min(dd, dlast)],
                    point_keys[pbase + min(dd + 1u, dlast)],
                    point_keys[pbase + min(dd + 2u, dlast)],
                    point_keys[pbase + min(dd + 3u, dlast)]);
                for (var j = 0u; j < 16u; j += 1u) {
                    acc[j] += dot(x, tile[j * 16u + p4]);
                }
            }
        }

        for (var j = 0u; j < 16u; j += 1u) {
            let ci = c0 + j;
            if (ci < c && acc[j] > best_sim) {
                best_sim = acc[j];
                best = ci;
            }
        }
    }

    if (valid) {
        assignments[row_off + local] = best;
    }
}
