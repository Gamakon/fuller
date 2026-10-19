@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 2048>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let t = lid.x;
    let valid = gid.x < n_band;
    var l = gid.x;
    if (!valid) { l = n_band - 1u; }
    let base = l * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        var s: array<f32, 16>;
        for (var r = 0u; r < 16u; r += 1u) { s[r] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let j = t + i * 128u;
                let row = j >> 7u;
                let col = j & 127u;
                let ci = c0 + row;
                let d = d0 + col;
                var v: f32 = 0.0;
                if (ci < c && d < dp) { v = centroid_keys[ci * dp + d]; }
                tile[j] = v;
            }
            workgroupBarrier();
            let kmax = min(128u, dp - d0);
            for (var k = 0u; k < kmax; k += 1u) {
                let pv = point_keys[base + d0 + k];
                for (var r = 0u; r < 16u; r += 1u) {
                    s[r] = s[r] + pv * tile[r * 128u + k];
                }
            }
        }

        for (var r = 0u; r < 16u; r += 1u) {
            let ci = c0 + r;
            if (ci < c && s[r] > best_sim) { best_sim = s[r]; best = ci; }
        }
    }

    if (valid) { assignments[row_off + gid.x] = best; }
}
