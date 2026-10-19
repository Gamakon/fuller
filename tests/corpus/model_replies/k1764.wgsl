@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let local = gid.x;
    let valid = local < n_band;
    var base = 0u;
    if (valid) { base = local * dp; }

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    let groups = (c + 15u) / 16u;
    for (var cg = 0u; cg < groups; cg += 1u) {
        var acc: array<f32, 16>;
        for (var j = 0u; j < 16u; j += 1u) { acc[j] = 0.0; }
        let c0 = cg * 16u;
        for (var d0 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            for (var i = 0u; i < 1024u; i += 128u) {
                let idx = i + lid;
                let ci = idx / 64u;
                let d = idx % 64u;
                let gc = c0 + ci;
                var v = 0.0;
                if (gc < c && d0 + d < dp) {
                    v = centroid_keys[gc * dp + d0 + d];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            let lim = min(64u, dp - d0);
            for (var k = 0u; k < lim; k += 1u) {
                let x = point_keys[base + d0 + k];
                for (var j = 0u; j < 16u; j += 1u) {
                    acc[j] = acc[j] + x * tile[j * 64u + k];
                }
            }
        }
        for (var j = 0u; j < 16u; j += 1u) {
            if (c0 + j < c) {
                let s = acc[j];
                if (s > best_sim) { best_sim = s; best = c0 + j; }
            }
        }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
