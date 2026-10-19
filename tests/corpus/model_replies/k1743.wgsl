@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 2048>;

fn upd(a: vec4<f32>, ci: u32, bs: ptr<function, vec4<f32>>, bi: ptr<function, vec4<u32>>) {
    let gt = a > *bs;
    *bs = select(*bs, a, gt);
    *bi = select(*bi, vec4<u32>(ci), gt);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;

    let l0 = wid.x * 256u + lid * 4u;
    let last = n_band - 1u;
    let b0 = min(l0, last) * dp;
    let b1 = min(l0 + 1u, last) * dp;
    let b2 = min(l0 + 2u, last) * dp;
    let b3 = min(l0 + 3u, last) * dp;

    var bs = vec4<f32>(-3.4e38);
    var bi = vec4<u32>(0u);

    for (var cb = 0u; cb < c; cb += 8u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        var a4 = vec4<f32>(0.0);
        var a5 = vec4<f32>(0.0);
        var a6 = vec4<f32>(0.0);
        var a7 = vec4<f32>(0.0);

        for (var kk = 0u; kk < dp; kk += 256u) {
            for (var i = 0u; i < 32u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx >> 8u;
                let k = idx & 255u;
                var v = 0.0;
                if (cb + cc < c && kk + k < dp) {
                    v = centroid_keys[(cb + cc) * dp + kk + k];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            let cnt = min(256u, dp - kk);
            for (var k = 0u; k < cnt; k += 1u) {
                let pv = vec4<f32>(
                    point_keys[b0 + kk + k],
                    point_keys[b1 + kk + k],
                    point_keys[b2 + kk + k],
                    point_keys[b3 + kk + k]);
                a0 += pv * tile[k];
                a1 += pv * tile[256u + k];
                a2 += pv * tile[512u + k];
                a3 += pv * tile[768u + k];
                a4 += pv * tile[1024u + k];
                a5 += pv * tile[1280u + k];
                a6 += pv * tile[1536u + k];
                a7 += pv * tile[1792u + k];
            }
            workgroupBarrier();
        }

        if (cb + 0u < c) { upd(a0, cb + 0u, &bs, &bi); }
        if (cb + 1u < c) { upd(a1, cb + 1u, &bs, &bi); }
        if (cb + 2u < c) { upd(a2, cb + 2u, &bs, &bi); }
        if (cb + 3u < c) { upd(a3, cb + 3u, &bs, &bi); }
        if (cb + 4u < c) { upd(a4, cb + 4u, &bs, &bi); }
        if (cb + 5u < c) { upd(a5, cb + 5u, &bs, &bi); }
        if (cb + 6u < c) { upd(a6, cb + 6u, &bs, &bi); }
        if (cb + 7u < c) { upd(a7, cb + 7u, &bs, &bi); }
    }

    if (l0 < n_band) { assignments[row_off + l0] = bi.x; }
    if (l0 + 1u < n_band) { assignments[row_off + l0 + 1u] = bi.y; }
    if (l0 + 2u < n_band) { assignments[row_off + l0 + 2u] = bi.z; }
    if (l0 + 3u < n_band) { assignments[row_off + l0 + 3u] = bi.w; }
}
