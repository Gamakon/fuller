// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    let ci = lidx & 63u;
    let it0 = lidx >> 6u;
    let gc = colBase + ci;
    let lrw = rowBase + ci;
    let grw = cfg.r_off + lrw;
    let okc = gc < d;
    let okr = (lrw < cfg.r_rows) && (grw < d);
    let tx = lid.x * 4u;
    let ty = lid.y * 4u;
    for (var t: u32 = 0u; t < m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let item = it0 + k * 4u;
            let tt = t + item;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (tt < m) {
                let base = tt * d;
                if (okc) { fb = phi[base + gc]; }
                if (okr) { fa = phi[base + grw]; }
            }
            let wi = (item * 64u + ci) & 511u;
            wg_a[wi] = fa;
            wg_b[wi] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let o = kk * 64u;
            let b = vec4<f32>(
                wg_b[(o + tx) & 511u],
                wg_b[(o + tx + 1u) & 511u],
                wg_b[(o + tx + 2u) & 511u],
                wg_b[(o + tx + 3u) & 511u]);
            let r0 = wg_a[(o + ty) & 511u];
            let r1 = wg_a[(o + ty + 1u) & 511u];
            let r2 = wg_a[(o + ty + 2u) & 511u];
            let r3 = wg_a[(o + ty + 3u) & 511u];
            a0 = fma(vec4<f32>(r0), b, a0);
            a1 = fma(vec4<f32>(r1), b, a1);
            a2 = fma(vec4<f32>(r2), b, a2);
            a3 = fma(vec4<f32>(r3), b, a3);
        }
        workgroupBarrier();
    }
    let c0 = colBase + tx;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + ty + j;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            var v = a0;
            if (j == 1u) { v = a1; }
            if (j == 2u) { v = a2; }
            if (j == 3u) { v = a3; }
            let rb = lr * d;
            if (c0 < d) { h[rb + c0] = h[rb + c0] + v.x; }
            if (c0 + 1u < d) { h[rb + c0 + 1u] = h[rb + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[rb + c0 + 2u] = h[rb + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[rb + c0 + 3u] = h[rb + c0 + 3u] + v.w; }
        }
    }
}
