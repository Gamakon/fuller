@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let l = gid.x;
    let valid = l < n_band;
    let lc = select(0u, l, valid);
    let pbase = lc * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 16u) {
        var acc: array<f32, 16>;
        for (var k: u32 = 0u; k < 16u; k += 1u) {
            acc[k] = 0.0;
        }
        for (var d0: u32 = 0u; d0 < dp; d0 += 64u) {
            workgroupBarrier();
            for (var i: u32 = 0u; i < 2u; i += 1u) {
                let t = lid + i * 128u;
                let cc = t / 16u;
                let q = t % 16u;
                let ci = cb + cc;
                let d = d0 + q * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let base = ci * dp + d;
                    if (d + 0u < dp) { v.x = centroid_keys[base]; }
                    if (d + 1u < dp) { v.y = centroid_keys[base + 1u]; }
                    if (d + 2u < dp) { v.z = centroid_keys[base + 2u]; }
                    if (d + 3u < dp) { v.w = centroid_keys[base + 3u]; }
                }
                tile[t] = v;
            }
            workgroupBarrier();
            for (var q: u32 = 0u; q < 16u; q += 1u) {
                let d = d0 + q * 4u;
                var p = vec4<f32>(0.0);
                let base = pbase + d;
                if (d + 0u < dp) { p.x = point_keys[base]; }
                if (d + 1u < dp) { p.y = point_keys[base + 1u]; }
                if (d + 2u < dp) { p.z = point_keys[base + 2u]; }
                if (d + 3u < dp) { p.w = point_keys[base + 3u]; }
                for (var cc: u32 = 0u; cc < 16u; cc += 1u) {
                    acc[cc] += dot(p, tile[cc * 16u + q]);
                }
            }
        }
        for (var cc: u32 = 0u; cc < 16u; cc += 1u) {
            let ci = cb + cc;
            if (ci < c && acc[cc] > best_sim) {
                best_sim = acc[cc];
                best = ci;
            }
        }
    }
    if (valid) {
        assignments[row_off + l] = best;
    }
}
