@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 512>;

fn ldp(base: u32, d: u32, dp: u32) -> vec4<f32> {
    if (d + 4u <= dp) {
        return vec4<f32>(point_keys[base + d], point_keys[base + d + 1u], point_keys[base + d + 2u], point_keys[base + d + 3u]);
    }
    var r = vec4<f32>(0.0);
    for (var j = 0u; j < 4u; j += 1u) {
        if (d + j < dp) { r[j] = point_keys[base + d + j]; }
    }
    return r;
}

fn ldc(base: u32, d: u32, dp: u32) -> vec4<f32> {
    if (d + 4u <= dp) {
        return vec4<f32>(centroid_keys[base + d], centroid_keys[base + d + 1u], centroid_keys[base + d + 2u], centroid_keys[base + d + 3u]);
    }
    var r = vec4<f32>(0.0);
    for (var j = 0u; j < 4u; j += 1u) {
        if (d + j < dp) { r[j] = centroid_keys[base + d + j]; }
    }
    return r;
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let l = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let row = min(l, n_band - 1u);
    let pbase = row * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        var a: array<f32, 16>;
        for (var j = 0u; j < 16u; j += 1u) { a[j] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 4u; i += 1u) {
                let s = i * 128u + lid;
                let cen = s / 32u;
                let v = s % 32u;
                let ci = c0 + cen;
                let d = d0 + v * 4u;
                var val = vec4<f32>(0.0);
                if (ci < c && d < dp) {
                    val = ldc(ci * dp, d, dp);
                }
                tile[s] = val;
            }
            workgroupBarrier();
            for (var k = 0u; k < 32u; k += 1u) {
                let d = d0 + k * 4u;
                let pv = ldp(pbase, d, dp);
                for (var j = 0u; j < 16u; j += 1u) {
                    a[j] += dot(pv, tile[j * 32u + k]);
                }
            }
        }

        for (var j = 0u; j < 16u; j += 1u) {
            let ci = c0 + j;
            if (ci < c && a[j] > best_sim) {
                best_sim = a[j];
                best = ci;
            }
        }
    }

    if (l < n_band) {
        assignments[row_off + l] = best;
    }
}
