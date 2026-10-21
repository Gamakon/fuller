// grid 32 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var e0: vec4<f32> = vec4<f32>(0.0);
    var e1: vec4<f32> = vec4<f32>(0.0);
    var e2: vec4<f32> = vec4<f32>(0.0);
    var e3: vec4<f32> = vec4<f32>(0.0);
    var o0: vec4<f32> = vec4<f32>(0.0);
    var o1: vec4<f32> = vec4<f32>(0.0);
    var o2: vec4<f32> = vec4<f32>(0.0);
    var o3: vec4<f32> = vec4<f32>(0.0);
    var s0: vec4<f32> = vec4<f32>(0.0);
    var s1: vec4<f32> = vec4<f32>(0.0);
    var s2: vec4<f32> = vec4<f32>(0.0);
    var s3: vec4<f32> = vec4<f32>(0.0);
    var k: u32 = 0u;
    var t: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let e = lidx + 256u * k;
            let ti = t + (e >> 5u);
            let col = wid.x * 128u + (e & 31u) * 4u;
            var v = vec4<f32>(0.0);
            if (ti < cfg.m) {
                let base = ti * cfg.d + col;
                if (col < cfg.d) { v.x = phi[base]; }
                if (col + 1u < cfg.d) { v.y = phi[base + 1u]; }
                if (col + 2u < cfg.d) { v.z = phi[base + 2u]; }
                if (col + 3u < cfg.d) { v.w = phi[base + 3u]; }
            }
            wg_va[e] = v;
        }
        if (lidx < 128u) {
            let ti = t + (lidx >> 3u);
            let lr = wid.y * 32u + (lidx & 7u) * 4u;
            let gr = cfg.r_off + lr;
            var w = vec4<f32>(0.0);
            if (ti < cfg.m) {
                let base = ti * cfg.d + gr;
                if (lr < cfg.r_rows && gr < cfg.d) { w.x = phi[base]; }
                if (lr + 1u < cfg.r_rows && gr + 1u < cfg.d) { w.y = phi[base + 1u]; }
                if (lr + 2u < cfg.r_rows && gr + 2u < cfg.d) { w.z = phi[base + 2u]; }
                if (lr + 3u < cfg.r_rows && gr + 3u < cfg.d) { w.w = phi[base + 3u]; }
            }
            wg_vb[lidx] = w;
        }
        workgroupBarrier();
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            let av = wg_va[k * 32u + lid.x];
            let bv = wg_vb[k * 8u + lid.y];
            a0 = a0 + vec4<f32>(bv.x) * av;
            a1 = a1 + vec4<f32>(bv.y) * av;
            a2 = a2 + vec4<f32>(bv.z) * av;
            a3 = a3 + vec4<f32>(bv.w) * av;
        }
        if (((t >> 4u) & 1u) == 0u) {
            e0 = e0 + a0; e1 = e1 + a1; e2 = e2 + a2; e3 = e3 + a3;
        } else {
            o0 = o0 + a0; o1 = o1 + a1; o2 = o2 + a2; o3 = o3 + a3;
        }
        workgroupBarrier();
    }
    s0 = e0 + o0;
    s1 = e1 + o1;
    s2 = e2 + o2;
    s3 = e3 + o3;
    var pv: array<vec4<f32>, 4>;
    pv[0u] = s0;
    pv[1u] = s1;
    pv[2u] = s2;
    pv[3u] = s3;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 32u + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 128u + lid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[i][j];
                }
            }
        }
    }
}
