@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;
var<workgroup> rs: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let pg = t & 15u;
    let cg = t >> 4u;
    let base_pt = wid.x * 64u + pg * 4u;

    var rowb: array<u32, 4>;
    var valid: array<bool, 4>;
    var bs: array<f32, 4>;
    var bi: array<u32, 4>;
    for (var k = 0u; k < 4u; k += 1u) {
        let l = base_pt + k;
        valid[k] = l < n;
        rowb[k] = min(l, max(n, 1u) - 1u) * dp;
        bs[k] = -3.4e38;
        bi[k] = 0u;
    }

    for (var ct = 0u; ct < c; ct += 16u) {
        var acc: array<vec4<f32>, 4>;
        for (var k = 0u; k < 4u; k += 1u) {
            acc[k] = vec4<f32>(0.0);
        }
        for (var dc = 0u; dc < dp; dc += 64u) {
            for (var i = 0u; i < 1024u; i += 64u) {
                let j = i + t;
                let cc = j >> 6u;
                let pp = j & 63u;
                let ci = ct + cc;
                let d = dc + pp;
                var v = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                tile[j] = v;
            }
            workgroupBarrier();
            let tb = cg * 256u;
            for (var p = 0u; p < 64u; p += 1u) {
                let d = min(dc + p, dp - 1u);
                let c0 = tile[tb + p];
                let c1 = tile[tb + 64u + p];
                let c2 = tile[tb + 128u + p];
                let c3 = tile[tb + 192u + p];
                let cv = vec4<f32>(c0, c1, c2, c3);
                for (var k = 0u; k < 4u; k += 1u) {
                    let pv = point_keys[rowb[k] + d];
                    acc[k] = acc[k] + cv * pv;
                }
            }
            workgroupBarrier();
        }
        for (var k = 0u; k < 4u; k += 1u) {
            for (var j = 0u; j < 4u; j += 1u) {
                let ci = ct + cg * 4u + j;
                if (ci < c && acc[k][j] > bs[k]) {
                    bs[k] = acc[k][j];
                    bi[k] = ci;
                }
            }
        }
    }

    for (var k = 0u; k < 4u; k += 1u) {
        let q = pg * 4u + k;
        rs[cg * 64u + q] = bs[k];
        ri[cg * 64u + q] = bi[k];
    }
    workgroupBarrier();
    let l = wid.x * 64u + t;
    if (l < n) {
        var best_s = rs[t];
        var best_i = ri[t];
        for (var g = 1u; g < 4u; g += 1u) {
            let s = rs[g * 64u + t];
            let ix = ri[g * 64u + t];
            if (s > best_s || (s == best_s && ix < best_i)) {
                best_s = s;
                best_i = ix;
            }
        }
        assignments[row_off + l] = best_i;
    }
}
