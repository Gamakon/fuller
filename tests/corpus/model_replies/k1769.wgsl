@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 512>;

fn pload(l: u32, p: u32, n: u32, dp: u32) -> vec4<f32> {
    if (l >= n) { return vec4<f32>(0.0); }
    let b = l * dp + p;
    if (p + 3u < dp) {
        return vec4<f32>(point_keys[b], point_keys[b + 1u], point_keys[b + 2u], point_keys[b + 3u]);
    }
    var r = vec4<f32>(0.0);
    for (var m = 0u; m < 4u; m += 1u) {
        if (p + m < dp) { r[m] = point_keys[b + m]; }
    }
    return r;
}

fn cload(ci: u32, p: u32, c: u32, dp: u32) -> vec4<f32> {
    if (ci >= c) { return vec4<f32>(0.0); }
    let b = ci * dp + p;
    if (p + 3u < dp) {
        return vec4<f32>(centroid_keys[b], centroid_keys[b + 1u], centroid_keys[b + 2u], centroid_keys[b + 3u]);
    }
    var r = vec4<f32>(0.0);
    for (var m = 0u; m < 4u; m += 1u) {
        if (p + m < dp) { r[m] = centroid_keys[b + m]; }
    }
    return r;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 256u + lid;

    var best_sim = array<f32, 4>(-3.4e38, -3.4e38, -3.4e38, -3.4e38);
    var best = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var ct = 0u; ct < c; ct += 8u) {
        var acc: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) { acc[i] = 0.0; }
        for (var d0 = 0u; d0 < dp; d0 += 256u) {
            workgroupBarrier();
            for (var i = 0u; i < 8u; i += 1u) {
                let v = lid + i * 64u;
                let k = v >> 6u;
                let q = v & 63u;
                let p = d0 + q * 4u;
                var t = vec4<f32>(0.0);
                if (p < dp) { t = cload(ct + k, p, c, dp); }
                tile[v] = t;
            }
            workgroupBarrier();
            for (var q = 0u; q < 64u; q += 1u) {
                let p = d0 + q * 4u;
                if (p < dp) {
                    let a0 = pload(base, p, n, dp);
                    let a1 = pload(base + 64u, p, n, dp);
                    let a2 = pload(base + 128u, p, n, dp);
                    let a3 = pload(base + 192u, p, n, dp);
                    for (var k = 0u; k < 8u; k += 1u) {
                        let t = tile[k * 64u + q];
                        acc[k] += dot(a0, t);
                        acc[8u + k] += dot(a1, t);
                        acc[16u + k] += dot(a2, t);
                        acc[24u + k] += dot(a3, t);
                    }
                }
            }
        }
        for (var k = 0u; k < 8u; k += 1u) {
            let ci = ct + k;
            if (ci < c) {
                for (var j = 0u; j < 4u; j += 1u) {
                    let s = acc[j * 8u + k];
                    if (s > best_sim[j]) { best_sim[j] = s; best[j] = ci; }
                }
            }
        }
    }

    for (var j = 0u; j < 4u; j += 1u) {
        let l = base + j * 64u;
        if (l < n) { assignments[row_off + l] = best[j]; }
    }
}
