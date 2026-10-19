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
    let valid = local < n_band;
    let lc = min(local, max(n_band, 1u) - 1u);
    let pbase = lc * dp;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 8u) {
        var acc: array<f32, 8>;
        for (var j = 0u; j < 8u; j += 1u) { acc[j] = 0.0; }

        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 256u; i += 64u) {
                let idx = i + lid;
                let cj = idx / 32u;
                let q = idx % 32u;
                let ci = cb + cj;
                let d = d0 + q * 4u;
                var v = vec4<f32>(0.0);
                if (ci < c) {
                    let base = ci * dp + d;
                    if (d < dp) { v.x = centroid_keys[base]; }
                    if (d + 1u < dp) { v.y = centroid_keys[base + 1u]; }
                    if (d + 2u < dp) { v.z = centroid_keys[base + 2u]; }
                    if (d + 3u < dp) { v.w = centroid_keys[base + 3u]; }
                }
                tile[idx] = v;
            }
            workgroupBarrier();

            for (var q = 0u; q < 32u; q += 1u) {
                let d = d0 + q * 4u;
                var pv = vec4<f32>(0.0);
                if (d < dp) { pv.x = point_keys[pbase + d]; }
                if (d + 1u < dp) { pv.y = point_keys[pbase + d + 1u]; }
                if (d + 2u < dp) { pv.z = point_keys[pbase + d + 2u]; }
                if (d + 3u < dp) { pv.w = point_keys[pbase + d + 3u]; }
                for (var j = 0u; j < 8u; j += 1u) {
                    let cv = tile[j * 32u + q];
                    acc[j] = acc[j] + dot(pv, cv);
                }
            }
        }

        for (var j = 0u; j < 8u; j += 1u) {
            let ci = cb + j;
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
