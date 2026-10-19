struct Best {
    sim: f32,
    idx: u32,
};

var<workgroup> tile: array<f32, 1024>;

fn upd(s: vec4<f32>, ci0: u32, c: u32, cur: Best) -> Best {
    var r = cur;
    if (ci0 < c && s.x > r.sim) { r.sim = s.x; r.idx = ci0; }
    if (ci0 + 1u < c && s.y > r.sim) { r.sim = s.y; r.idx = ci0 + 1u; }
    if (ci0 + 2u < c && s.z > r.sim) { r.sim = s.z; r.idx = ci0 + 2u; }
    if (ci0 + 3u < c && s.w > r.sim) { r.sim = s.w; r.idx = ci0 + 3u; }
    return r;
}

@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 256u + lid;
    let l0 = base;
    let l1 = base + 64u;
    let l2 = base + 128u;
    let l3 = base + 192u;
    let last = max(n, 1u) - 1u;
    let o0 = min(l0, last) * dp;
    let o1 = min(l1, last) * dp;
    let o2 = min(l2, last) * dp;
    let o3 = min(l3, last) * dp;

    var b0 = Best(-3.4e38, 0u);
    var b1 = Best(-3.4e38, 0u);
    var b2 = Best(-3.4e38, 0u);
    var b3 = Best(-3.4e38, 0u);

    for (var ci0 = 0u; ci0 < c; ci0 += 4u) {
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < dp; d0 += 256u) {
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx >> 8u;
                let d = idx & 255u;
                var v = 0.0;
                if (ci0 + cc < c && d0 + d < dp) {
                    v = centroid_keys[(ci0 + cc) * dp + d0 + d];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            let len = min(256u, dp - d0);
            for (var dd = 0u; dd < 256u; dd += 1u) {
                if (dd >= len) { break; }
                let cv = vec4<f32>(tile[dd], tile[256u + dd], tile[512u + dd], tile[768u + dd]);
                let g = d0 + dd;
                s0 += point_keys[o0 + g] * cv;
                s1 += point_keys[o1 + g] * cv;
                s2 += point_keys[o2 + g] * cv;
                s3 += point_keys[o3 + g] * cv;
            }
        }
        b0 = upd(s0, ci0, c, b0);
        b1 = upd(s1, ci0, c, b1);
        b2 = upd(s2, ci0, c, b2);
        b3 = upd(s3, ci0, c, b3);
    }

    if (l0 < n) { assignments[row_off + l0] = b0.idx; }
    if (l1 < n) { assignments[row_off + l1] = b1.idx; }
    if (l2 < n) { assignments[row_off + l2] = b2.idx; }
    if (l3 < n) { assignments[row_off + l3] = b3.idx; }
}
