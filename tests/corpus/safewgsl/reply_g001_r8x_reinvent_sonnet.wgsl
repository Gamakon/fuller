// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let rowBase = wid.y * 64u;
    let colBase = wid.x * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 256u * k;
            let tt = e >> 6u;
            let cc = e & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t + tt < m) {
                let base = (t + tt) * d;
                let lrr = rowBase + cc;
                if (lrr < cfg.r_rows && cfg.r_off + lrr < d) {
                    fa = phi[base + cfg.r_off + lrr];
                }
                if (colBase + cc < d) {
                    fb = phi[base + colBase + cc];
                }
            }
            wg_a[e] = fa;
            wg_b[e] = fb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let bv = vec4<f32>(wg_b[o + lid.x * 4u], wg_b[o + lid.x * 4u + 1u], wg_b[o + lid.x * 4u + 2u], wg_b[o + lid.x * 4u + 3u]);
            let ra = o + lid.y * 4u;
            a0 = a0 + wg_a[ra] * bv;
            a1 = a1 + wg_a[ra + 1u] * bv;
            a2 = a2 + wg_a[ra + 2u] * bv;
            a3 = a3 + wg_a[ra + 3u] * bv;
        }
        workgroupBarrier();
    }
    let c0 = colBase + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let idx = lr * d + c0;
            if (c0 < d) { h[idx] = h[idx] + v.x; }
            if (c0 + 1u < d) { h[idx + 1u] = h[idx + 1u] + v.y; }
            if (c0 + 2u < d) { h[idx + 2u] = h[idx + 2u] + v.z; }
            if (c0 + 3u < d) { h[idx + 3u] = h[idx + 3u] + v.w; }
        }
    }
}
