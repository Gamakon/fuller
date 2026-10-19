@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 2048>;

fn pk(base: u32, d: u32, dp: u32) -> f32 {
    return select(0.0, point_keys[base + d], d < dp);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;

    let la = wid.x * 128u + lid;
    let lb = la + 64u;
    let va = la < n_band;
    let vb = lb < n_band;
    let last = max(n_band, 1u) - 1u;
    let baseA = min(la, last) * dp;
    let baseB = min(lb, last) * dp;

    var bestA: u32 = 0u;
    var bestB: u32 = 0u;
    var simA: f32 = -3.4e38;
    var simB: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += 16u) {
        var sa: array<f32, 16>;
        var sb: array<f32, 16>;
        for (var q = 0u; q < 16u; q += 1u) {
            sa[q] = 0.0;
            sb[q] = 0.0;
        }
        for (var d0: u32 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 32u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx >> 7u;
                let dd = idx & 127u;
                let ci = cb + cc;
                let d = d0 + dd;
                var v: f32 = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 32u; k += 1u) {
                let d = d0 + k * 4u;
                let pa = vec4<f32>(pk(baseA, d, dp), pk(baseA, d + 1u, dp), pk(baseA, d + 2u, dp), pk(baseA, d + 3u, dp));
                let pb = vec4<f32>(pk(baseB, d, dp), pk(baseB, d + 1u, dp), pk(baseB, d + 2u, dp), pk(baseB, d + 3u, dp));
                for (var q = 0u; q < 16u; q += 1u) {
                    let o = q * 128u + k * 4u;
                    let t = vec4<f32>(tile[o], tile[o + 1u], tile[o + 2u], tile[o + 3u]);
                    sa[q] += dot(pa, t);
                    sb[q] += dot(pb, t);
                }
            }
        }
        for (var q = 0u; q < 16u; q += 1u) {
            let ci = cb + q;
            if (ci < c) {
                if (sa[q] > simA) { simA = sa[q]; bestA = ci; }
                if (sb[q] > simB) { simB = sb[q]; bestB = ci; }
            }
        }
    }

    if (va) { assignments[row_off + la] = bestA; }
    if (vb) { assignments[row_off + lb] = bestB; }
}
