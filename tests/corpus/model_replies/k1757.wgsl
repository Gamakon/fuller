@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let l = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let dv = dp / 4u;
    let valid = l < n_band;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    let pbase = l * dp;

    for (var cb = 0u; cb < c; cb += 16u) {
        var sums: array<f32, 16>;
        for (var k = 0u; k < 16u; k += 1u) {
            sums[k] = 0.0;
        }
        for (var dc = 0u; dc < dv; dc += 16u) {
            workgroupBarrier();
            for (var i = 0u; i < 4u; i += 1u) {
                let idx = lid + i * 64u;
                let cc = idx >> 4u;
                let j = idx & 15u;
                let ci = cb + cc;
                let dj = dc + j;
                var v = vec4<f32>(0.0);
                if (ci < c && dj < dv) {
                    let o = ci * dp + dj * 4u;
                    v = vec4<f32>(centroid_keys[o], centroid_keys[o + 1u], centroid_keys[o + 2u], centroid_keys[o + 3u]);
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var j = 0u; j < 16u; j += 1u) {
                let dj = dc + j;
                var pv = vec4<f32>(0.0);
                if (valid && dj < dv) {
                    let o = pbase + dj * 4u;
                    pv = vec4<f32>(point_keys[o], point_keys[o + 1u], point_keys[o + 2u], point_keys[o + 3u]);
                }
                for (var k = 0u; k < 16u; k += 1u) {
                    sums[k] += dot(pv, tile[k * 16u + j]);
                }
            }
        }
        for (var k = 0u; k < 16u; k += 1u) {
            if (cb + k < c && sums[k] > best_sim) {
                best_sim = sums[k];
                best = cb + k;
            }
        }
    }
    if (valid) {
        assignments[row_off + l] = best;
    }
}
