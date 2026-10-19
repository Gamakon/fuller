// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let item = lidx >> 4u;
    let c4 = (lidx & 15u) * 4u;
    let colb = wid.x * 64u + c4;
    let lrb = wid.y * 64u + c4;
    let grb = cfg.r_off + lrb;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        let ti = t + item;
        var vb = vec4<f32>(0.0);
        var va = vec4<f32>(0.0);
        if (ti < m) {
            let base = ti * d;
            if (colb < d) { vb.x = phi[base + colb]; }
            if (colb + 1u < d) { vb.y = phi[base + colb + 1u]; }
            if (colb + 2u < d) { vb.z = phi[base + colb + 2u]; }
            if (colb + 3u < d) { vb.w = phi[base + colb + 3u]; }
            if (lrb < cfg.r_rows && grb < d) { va.x = phi[base + grb]; }
            if (lrb + 1u < cfg.r_rows && grb + 1u < d) { va.y = phi[base + grb + 1u]; }
            if (lrb + 2u < cfg.r_rows && grb + 2u < d) { va.z = phi[base + grb + 2u]; }
            if (lrb + 3u < cfg.r_rows && grb + 3u < d) { va.w = phi[base + grb + 3u]; }
        }
        sa[lidx & 255u] = va;
        sb[lidx & 255u] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = sa[(k * 16u + lid.y) & 255u];
            let b = sb[(k * 16u + lid.x) & 255u];
            a0 = fma(vec4<f32>(a.x), b, a0);
            a1 = fma(vec4<f32>(a.y), b, a1);
            a2 = fma(vec4<f32>(a.z), b, a2);
            a3 = fma(vec4<f32>(a.w), b, a3);
        }
        workgroupBarrier();
    }
    let c0 = wid.x * 64u + lid.x * 4u;
    let r0 = wid.y * 64u + lid.y * 4u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = r0 + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = a0;
            if (j == 1u) { v = a1; }
            if (j == 2u) { v = a2; }
            if (j == 3u) { v = a3; }
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + v.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
