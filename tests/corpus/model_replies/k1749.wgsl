@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

fn ldc(ci: u32, p: u32, c: u32, dp: u32) -> f32 {
    if (ci < c && p < dp) {
        return centroid_keys[ci * dp + p];
    }
    return 0.0;
}

fn ldp(l: u32, p: u32, n: u32, dp: u32) -> f32 {
    if (l < n && p < dp) {
        return point_keys[l * dp + p];
    }
    return 0.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var ct = 0u; ct < c; ct += 16u) {
        var acc: array<f32, 16>;
        for (var j = 0u; j < 16u; j += 1u) {
            acc[j] = 0.0;
        }
        for (var dt = 0u; dt < dp; dt += 64u) {
            workgroupBarrier();
            for (var i = 0u; i < 4u; i += 1u) {
                let v = lid + 64u * i;
                let cj = v / 16u;
                let k = v % 16u;
                let gci = ct + cj;
                let gp = dt + 4u * k;
                tile[v] = vec4<f32>(
                    ldc(gci, gp, c, dp),
                    ldc(gci, gp + 1u, c, dp),
                    ldc(gci, gp + 2u, c, dp),
                    ldc(gci, gp + 3u, c, dp));
            }
            workgroupBarrier();
            for (var k = 0u; k < 16u; k += 1u) {
                let gp = dt + 4u * k;
                let pv = vec4<f32>(
                    ldp(local, gp, n_band, dp),
                    ldp(local, gp + 1u, n_band, dp),
                    ldp(local, gp + 2u, n_band, dp),
                    ldp(local, gp + 3u, n_band, dp));
                for (var j = 0u; j < 16u; j += 1u) {
                    acc[j] += dot(pv, tile[j * 16u + k]);
                }
            }
        }
        for (var j = 0u; j < 16u; j += 1u) {
            let ci = ct + j;
            if (ci < c && acc[j] > best_sim) {
                best_sim = acc[j];
                best = ci;
            }
        }
    }

    if (local < n_band) {
        assignments[row_off + local] = best;
    }
}
