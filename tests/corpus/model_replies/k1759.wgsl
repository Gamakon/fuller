@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> ctile: array<f32, 512>;
var<workgroup> ms: array<f32, 256>;
var<workgroup> mi: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = params.x;
    let C = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pq = lid % 16u;
    let cl = lid / 16u;
    let base_pt = wid.x * 64u + pq * 4u;

    var pb: array<u32, 4>;
    for (var j = 0u; j < 4u; j += 1u) {
        let l = min(base_pt + j, max(n, 1u) - 1u);
        pb[j] = l * dp;
    }

    var bs: array<f32, 4>;
    var bi: array<u32, 4>;
    for (var j = 0u; j < 4u; j += 1u) {
        bs[j] = -3.4e38;
        bi[j] = 0xffffffffu;
    }

    for (var c0 = 0u; c0 < C; c0 += 16u) {
        var s: array<f32, 16>;
        for (var q = 0u; q < 16u; q += 1u) { s[q] = 0.0; }

        for (var d0 = 0u; d0 < dp; d0 += 32u) {
            workgroupBarrier();
            for (var i = 0u; i < 8u; i += 1u) {
                let idx = i * 64u + lid;
                let c = idx / 32u;
                let k = idx % 32u;
                let ci = c0 + c;
                let d = d0 + k;
                var v = 0.0;
                if (ci < C && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                ctile[idx] = v;
            }
            workgroupBarrier();

            for (var k = 0u; k < 32u; k += 1u) {
                let d = d0 + k;
                var pv: array<f32, 4>;
                if (d < dp) {
                    for (var j = 0u; j < 4u; j += 1u) {
                        pv[j] = point_keys[pb[j] + d];
                    }
                } else {
                    for (var j = 0u; j < 4u; j += 1u) { pv[j] = 0.0; }
                }
                for (var cc = 0u; cc < 4u; cc += 1u) {
                    let w = ctile[(cl * 4u + cc) * 32u + k];
                    for (var j = 0u; j < 4u; j += 1u) {
                        s[cc * 4u + j] = s[cc * 4u + j] + w * pv[j];
                    }
                }
            }
        }

        for (var cc = 0u; cc < 4u; cc += 1u) {
            let ci = c0 + cl * 4u + cc;
            if (ci < C) {
                for (var j = 0u; j < 4u; j += 1u) {
                    let sim = s[cc * 4u + j];
                    if (sim > bs[j]) {
                        bs[j] = sim;
                        bi[j] = ci;
                    }
                }
            }
        }
    }

    for (var j = 0u; j < 4u; j += 1u) {
        let slot = cl * 64u + pq * 4u + j;
        ms[slot] = bs[j];
        mi[slot] = bi[j];
    }
    workgroupBarrier();

    let l = wid.x * 64u + lid;
    if (l < n) {
        var best_s = ms[lid];
        var best_i = mi[lid];
        for (var q = 1u; q < 4u; q += 1u) {
            let sv = ms[q * 64u + lid];
            let iv = mi[q * 64u + lid];
            if (sv > best_s || (sv == best_s && iv < best_i)) {
                best_s = sv;
                best_i = iv;
            }
        }
        assignments[row_off + l] = best_i;
    }
}
