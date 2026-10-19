@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<vec4<f32>, 512>;
var<workgroup> mbs: array<f32, 256>;
var<workgroup> mbi: array<u32, 256>;

fn ld4(o: u32) -> vec4<f32> {
    return vec4<f32>(point_keys[o], point_keys[o + 1u], point_keys[o + 2u], point_keys[o + 3u]);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let dp4 = dp / 4u;
    let pg = lid % 16u;
    let cs = lid / 16u;
    let lbase = wid.x * 64u + pg * 4u;

    var o0 = min(lbase, n - 1u) * dp;
    var o1 = min(lbase + 1u, n - 1u) * dp;
    var o2 = min(lbase + 2u, n - 1u) * dp;
    var o3 = min(lbase + 3u, n - 1u) * dp;

    var bs = array<f32, 4>(-3.4e38, -3.4e38, -3.4e38, -3.4e38);
    var bi = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var cb = 0u; cb < c; cb += 16u) {
        var s = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (var d0 = 0u; d0 < dp4; d0 += 32u) {
            for (var k = 0u; k < 8u; k += 1u) {
                let idx = lid + k * 64u;
                let cl = idx / 32u;
                let dv = idx % 32u;
                let ci = min(cb + cl, c - 1u);
                let d = d0 + dv;
                var v = vec4<f32>(0.0);
                if (d < dp4) {
                    let o = ci * dp + d * 4u;
                    v = vec4<f32>(centroid_keys[o], centroid_keys[o + 1u], centroid_keys[o + 2u], centroid_keys[o + 3u]);
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            for (var dv = 0u; dv < 32u; dv += 1u) {
                let d = d0 + dv;
                if (d < dp4) {
                    let p0 = ld4(o0 + d * 4u);
                    let p1 = ld4(o1 + d * 4u);
                    let p2 = ld4(o2 + d * 4u);
                    let p3 = ld4(o3 + d * 4u);
                    for (var j = 0u; j < 4u; j += 1u) {
                        let cv = tile[(cs * 4u + j) * 32u + dv];
                        s[j * 4u + 0u] += dot(p0, cv);
                        s[j * 4u + 1u] += dot(p1, cv);
                        s[j * 4u + 2u] += dot(p2, cv);
                        s[j * 4u + 3u] += dot(p3, cv);
                    }
                }
            }
            workgroupBarrier();
        }
        for (var j = 0u; j < 4u; j += 1u) {
            let ci = cb + cs * 4u + j;
            if (ci < c) {
                for (var q = 0u; q < 4u; q += 1u) {
                    let sim = s[j * 4u + q];
                    if (sim > bs[q]) {
                        bs[q] = sim;
                        bi[q] = ci;
                    }
                }
            }
        }
    }

    for (var q = 0u; q < 4u; q += 1u) {
        mbs[lid * 4u + q] = bs[q];
        mbi[lid * 4u + q] = bi[q];
    }
    workgroupBarrier();
    if (cs == 0u) {
        for (var q = 0u; q < 4u; q += 1u) {
            var best = bs[q];
            var bidx = bi[q];
            for (var sl = 1u; sl < 4u; sl += 1u) {
                let m = (sl * 16u + pg) * 4u + q;
                let sv = mbs[m];
                let si = mbi[m];
                if (sv > best || (sv == best && si < bidx)) {
                    best = sv;
                    bidx = si;
                }
            }
            let l = lbase + q;
            if (l < n) {
                assignments[row_off + l] = bidx;
            }
        }
    }
}
