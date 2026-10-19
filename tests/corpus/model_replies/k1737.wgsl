@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> pt: array<vec4<f32>, 512>;
var<workgroup> ct: array<vec4<f32>, 128>;

fn upd(bs: ptr<function, f32>, bi: ptr<function, u32>, s: f32, ci: u32, c: u32) {
    if (ci < c && s > *bs) {
        *bs = s;
        *bi = ci;
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let tx = t & 7u;
    let ty = t >> 3u;

    for (var pb = 0u; pb < 2u; pb += 1u) {
        let p0 = wid.x * 256u + pb * 128u;
        var s0: f32 = -3.4e38; var s1: f32 = -3.4e38; var s2: f32 = -3.4e38; var s3: f32 = -3.4e38;
        var i0: u32 = 0xFFFFFFFFu; var i1: u32 = 0xFFFFFFFFu; var i2: u32 = 0xFFFFFFFFu; var i3: u32 = 0xFFFFFFFFu;

        for (var c0 = 0u; c0 < c; c0 += 32u) {
            var a0 = vec4<f32>(0.0);
            var a1 = vec4<f32>(0.0);
            var a2 = vec4<f32>(0.0);
            var a3 = vec4<f32>(0.0);
            for (var d0 = 0u; d0 < dp; d0 += 16u) {
                workgroupBarrier();
                let prow = t >> 1u;
                let pkq = (t & 1u) * 8u;
                let gl = p0 + prow;
                for (var i = 0u; i < 8u; i += 1u) {
                    let d = d0 + pkq + i;
                    var v: f32 = 0.0;
                    if (gl < n && d < dp) {
                        v = point_keys[gl * dp + d];
                    }
                    pt[(pkq + i) * 32u + (prow >> 2u)][prow & 3u] = v;
                }
                let crow = t >> 3u;
                let ckq = (t & 7u) * 2u;
                let cg = c0 + crow;
                for (var i = 0u; i < 2u; i += 1u) {
                    let d = d0 + ckq + i;
                    var v: f32 = 0.0;
                    if (cg < c && d < dp) {
                        v = centroid_keys[cg * dp + d];
                    }
                    ct[(ckq + i) * 8u + (crow >> 2u)][crow & 3u] = v;
                }
                workgroupBarrier();
                for (var k = 0u; k < 16u; k += 1u) {
                    let pv = pt[k * 32u + ty];
                    let cv = ct[k * 8u + tx];
                    a0 = a0 + pv.x * cv;
                    a1 = a1 + pv.y * cv;
                    a2 = a2 + pv.z * cv;
                    a3 = a3 + pv.w * cv;
                }
            }
            let cb = c0 + tx * 4u;
            upd(&s0, &i0, a0.x, cb, c); upd(&s0, &i0, a0.y, cb + 1u, c);
            upd(&s0, &i0, a0.z, cb + 2u, c); upd(&s0, &i0, a0.w, cb + 3u, c);
            upd(&s1, &i1, a1.x, cb, c); upd(&s1, &i1, a1.y, cb + 1u, c);
            upd(&s1, &i1, a1.z, cb + 2u, c); upd(&s1, &i1, a1.w, cb + 3u, c);
            upd(&s2, &i2, a2.x, cb, c); upd(&s2, &i2, a2.y, cb + 1u, c);
            upd(&s2, &i2, a2.z, cb + 2u, c); upd(&s2, &i2, a2.w, cb + 3u, c);
            upd(&s3, &i3, a3.x, cb, c); upd(&s3, &i3, a3.y, cb + 1u, c);
            upd(&s3, &i3, a3.z, cb + 2u, c); upd(&s3, &i3, a3.w, cb + 3u, c);
        }

        workgroupBarrier();
        let r0 = (ty * 4u) * 8u + tx;
        // sims at flat [0,1024), idx at flat [1024,2048)
        for (var i = 0u; i < 4u; i += 1u) {
            var sv = s0;
            var iv = i0;
            if (i == 1u) { sv = s1; iv = i1; }
            if (i == 2u) { sv = s2; iv = i2; }
            if (i == 3u) { sv = s3; iv = i3; }
            let slot = r0 + i * 8u;
            pt[slot >> 2u][slot & 3u] = sv;
            let slot2 = slot + 1024u;
            pt[slot2 >> 2u][slot2 & 3u] = bitcast<f32>(iv);
        }
        workgroupBarrier();
        if (t < 128u) {
            var bs: f32 = -3.4e38;
            var bi: u32 = 0xFFFFFFFFu;
            for (var j = 0u; j < 8u; j += 1u) {
                let slot = t * 8u + j;
                let sv = pt[slot >> 2u][slot & 3u];
                let slot2 = slot + 1024u;
                let iv = bitcast<u32>(pt[slot2 >> 2u][slot2 & 3u]);
                if (sv > bs || (sv == bs && iv < bi)) {
                    bs = sv;
                    bi = iv;
                }
            }
            let l = p0 + t;
            if (l < n) {
                assignments[row_off + l] = select(bi, 0u, bi == 0xFFFFFFFFu);
            }
        }
        workgroupBarrier();
    }
}
