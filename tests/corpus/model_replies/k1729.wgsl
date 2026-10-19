@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> ps: array<f32, 2112>;
var<workgroup> cs: array<f32, 512>;
var<workgroup> bs: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 64u;
    let pg = t / 4u;
    let cl = t % 4u;

    var best_s: array<f32, 4>;
    var best_i: array<u32, 4>;
    for (var k = 0u; k < 4u; k++) {
        best_s[k] = -3.4e38;
        best_i[k] = 0u;
    }

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        var acc: array<f32, 16>;
        for (var q = 0u; q < 16u; q++) { acc[q] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 32u) {
            workgroupBarrier();
            for (var i = 0u; i < 32u; i++) {
                let idx = i * 64u + t;
                let row = idx / 32u;
                let col = idx % 32u;
                let l = base + row;
                var v = 0.0;
                if (l < n_band && d0 + col < dp) {
                    v = point_keys[l * dp + d0 + col];
                }
                ps[row * 33u + col] = v;
            }
            for (var i = 0u; i < 8u; i++) {
                let idx = i * 64u + t;
                let row = idx / 32u;
                let col = idx % 32u;
                let ci = c0 + row;
                var v = 0.0;
                if (ci < c && d0 + col < dp) {
                    v = centroid_keys[ci * dp + d0 + col];
                }
                cs[idx] = v;
            }
            workgroupBarrier();
            for (var d = 0u; d < 32u; d++) {
                let p0 = ps[(pg * 4u) * 33u + d];
                let p1 = ps[(pg * 4u + 1u) * 33u + d];
                let p2 = ps[(pg * 4u + 2u) * 33u + d];
                let p3 = ps[(pg * 4u + 3u) * 33u + d];
                for (var j = 0u; j < 4u; j++) {
                    let cv = cs[(cl + 4u * j) * 32u + d];
                    acc[j * 4u] += p0 * cv;
                    acc[j * 4u + 1u] += p1 * cv;
                    acc[j * 4u + 2u] += p2 * cv;
                    acc[j * 4u + 3u] += p3 * cv;
                }
            }
        }

        for (var j = 0u; j < 4u; j++) {
            let ci = c0 + cl + 4u * j;
            if (ci < c) {
                for (var k = 0u; k < 4u; k++) {
                    let s = acc[j * 4u + k];
                    if (s > best_s[k]) {
                        best_s[k] = s;
                        best_i[k] = ci;
                    }
                }
            }
        }
    }

    for (var k = 0u; k < 4u; k++) {
        bs[t * 4u + k] = best_s[k];
        bi[t * 4u + k] = best_i[k];
    }
    workgroupBarrier();
    if (cl == 0u) {
        for (var k = 0u; k < 4u; k++) {
            var s = bs[t * 4u + k];
            var ix = bi[t * 4u + k];
            for (var m = 1u; m < 4u; m++) {
                let s2 = bs[(t + m) * 4u + k];
                let i2 = bi[(t + m) * 4u + k];
                if (s2 > s || (s2 == s && i2 < ix)) {
                    s = s2;
                    ix = i2;
                }
            }
            let l = base + pg * 4u + k;
            if (l < n_band) {
                assignments[row_off + l] = ix;
            }
        }
    }
}
