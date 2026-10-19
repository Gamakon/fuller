@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let l = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = l < n_band;
    let base = select(0u, l * dp, valid);

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 16u) {
        var s: array<f32, 16>;
        for (var j = 0u; j < 16u; j += 1u) { s[j] = 0.0; }

        for (var pd: u32 = 0u; pd < dp; pd += 64u) {
            workgroupBarrier();
            let dd = pd + lid;
            for (var k = 0u; k < 16u; k += 1u) {
                let ci = cb + k;
                var v: f32 = 0.0;
                if (ci < c && dd < dp) {
                    v = centroid_keys[ci * dp + dd];
                }
                tile[k * 64u + lid] = v;
            }
            workgroupBarrier();

            for (var d = 0u; d < 64u; d += 1u) {
                let pidx = pd + d;
                var pv: f32 = 0.0;
                if (valid && pidx < dp) {
                    pv = point_keys[base + pidx];
                }
                for (var j = 0u; j < 16u; j += 1u) {
                    s[j] = s[j] + pv * tile[j * 64u + d];
                }
            }
        }

        for (var j = 0u; j < 16u; j += 1u) {
            if (cb + j < c) {
                if (s[j] > best_sim) { best_sim = s[j]; best = cb + j; }
            }
        }
    }

    if (valid) {
        assignments[row_off + l] = best;
    }
}
