@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let dp4 = dp / 4u;
    let valid = local < n_band;
    let pbase = select(0u, local * dp, valid);

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cg = 0u; cg < c; cg += 8u) {
        var acc: array<vec4<f32>, 8>;
        for (var j = 0u; j < 8u; j += 1u) {
            acc[j] = vec4<f32>(0.0);
        }
        for (var d = 0u; d < dp4; d += 32u) {
            workgroupBarrier();
            for (var r = 0u; r < 4u; r += 1u) {
                let e = r * 64u + lid;
                let j = e / 32u;
                let k = e % 32u;
                let ci = cg + j;
                let dd = d + k;
                var v = vec4<f32>(0.0);
                if (ci < c && dd < dp4) {
                    let b = ci * dp + dd * 4u;
                    v = vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
                }
                tile[e] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 32u; k += 1u) {
                let dd = d + k;
                if (dd < dp4) {
                    let b = pbase + dd * 4u;
                    let pv = vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
                    for (var j = 0u; j < 8u; j += 1u) {
                        acc[j] = acc[j] + pv * tile[j * 32u + k];
                    }
                }
            }
        }
        for (var j = 0u; j < 8u; j += 1u) {
            let ci = cg + j;
            if (ci < c) {
                var sim = acc[j].x + acc[j].y + acc[j].z + acc[j].w;
                for (var q = 0u; q < 3u; q += 1u) {
                    let p = dp4 * 4u + q;
                    if (p < dp && valid) {
                        sim = sim + point_keys[pbase + p] * centroid_keys[ci * dp + p];
                    }
                }
                if (sim > best_sim) { best_sim = sim; best = ci; }
            }
        }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
