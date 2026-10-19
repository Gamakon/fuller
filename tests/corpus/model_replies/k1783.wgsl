@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> sh: array<f32, 2048>;
var<workgroup> msim: array<f32, 256>;
var<workgroup> midx: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pg = t & 15u;
    let s = t >> 4u;
    let base = wid.x * 64u + pg * 4u;
    let last = n_band - 1u;
    let o0 = min(base, last) * dp;
    let o1 = min(base + 1u, last) * dp;
    let o2 = min(base + 2u, last) * dp;
    let o3 = min(base + 3u, last) * dp;

    var best_sim = vec4<f32>(-3.4e38);
    var best_idx = vec4<u32>(0u);

    for (var cb = 0u; cb < c; cb += 16u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var dt = 0u; dt < dp; dt += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 32u; i += 1u) {
                let e = i * 64u + t;
                let j = e >> 7u;
                let d = e & 127u;
                let ci = cb + j;
                let dd = dt + d;
                var v = 0.0;
                if (ci < c && dd < dp) {
                    v = centroid_keys[ci * dp + dd];
                }
                sh[e] = v;
            }
            workgroupBarrier();
            let len = min(128u, dp - dt);
            let sb = s * 512u;
            for (var d = 0u; d < len; d += 1u) {
                let k = dt + d;
                let a = vec4<f32>(point_keys[o0 + k], point_keys[o1 + k], point_keys[o2 + k], point_keys[o3 + k]);
                a0 += a * sh[sb + d];
                a1 += a * sh[sb + 128u + d];
                a2 += a * sh[sb + 256u + d];
                a3 += a * sh[sb + 384u + d];
            }
        }
        let c0 = cb + s * 4u;
        if (c0 < c) {
            let g = a0 > best_sim;
            best_sim = select(best_sim, a0, g);
            best_idx = select(best_idx, vec4<u32>(c0), g);
        }
        if (c0 + 1u < c) {
            let g = a1 > best_sim;
            best_sim = select(best_sim, a1, g);
            best_idx = select(best_idx, vec4<u32>(c0 + 1u), g);
        }
        if (c0 + 2u < c) {
            let g = a2 > best_sim;
            best_sim = select(best_sim, a2, g);
            best_idx = select(best_idx, vec4<u32>(c0 + 2u), g);
        }
        if (c0 + 3u < c) {
            let g = a3 > best_sim;
            best_sim = select(best_sim, a3, g);
            best_idx = select(best_idx, vec4<u32>(c0 + 3u), g);
        }
    }

    let slot = s * 64u + pg * 4u;
    msim[slot] = best_sim.x;
    msim[slot + 1u] = best_sim.y;
    msim[slot + 2u] = best_sim.z;
    msim[slot + 3u] = best_sim.w;
    midx[slot] = best_idx.x;
    midx[slot + 1u] = best_idx.y;
    midx[slot + 2u] = best_idx.z;
    midx[slot + 3u] = best_idx.w;
    workgroupBarrier();

    if (t < 16u) {
        for (var i = 0u; i < 4u; i += 1u) {
            let pl = t * 4u + i;
            var bs = msim[pl];
            var bi = midx[pl];
            for (var q = 1u; q < 4u; q += 1u) {
                let sv = msim[q * 64u + pl];
                let iv = midx[q * 64u + pl];
                if (sv > bs || (sv == bs && iv < bi)) {
                    bs = sv;
                    bi = iv;
                }
            }
            let l = wid.x * 64u + pl;
            if (l < n_band) {
                assignments[row_off + l] = bi;
            }
        }
    }
}
