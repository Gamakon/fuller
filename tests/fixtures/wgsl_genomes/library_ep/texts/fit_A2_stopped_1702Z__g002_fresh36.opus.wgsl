// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 1024>;
var<workgroup> wb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let cbase = wid.x * 32u;
    let rbase = wid.y * 32u;
    var s00: f32 = 0.0; var s01: f32 = 0.0; var s10: f32 = 0.0; var s11: f32 = 0.0;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = lidx + 256u * i;
            let it = t + (idx >> 5u);
            let off = idx & 31u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < m) {
                let ca = cbase + off;
                if (ca < d) { va = phi[it * d + ca]; }
                let lr = rbase + off;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { vb = phi[it * d + gr]; }
            }
            wa[idx] = va;
            wb[idx] = vb;
        }
        workgroupBarrier();
        var p00: f32 = 0.0; var p01: f32 = 0.0; var p10: f32 = 0.0; var p11: f32 = 0.0;
        var q00: f32 = 0.0; var q01: f32 = 0.0; var q10: f32 = 0.0; var q11: f32 = 0.0;
        for (var k: u32 = 0u; k < 32u; k = k + 2u) {
            let o = k * 32u;
            let a0 = wa[o + lid.x];
            let a1 = wa[o + lid.x + 16u];
            let b0 = wb[o + lid.y];
            let b1 = wb[o + lid.y + 16u];
            p00 = p00 + b0 * a0; p01 = p01 + b0 * a1;
            p10 = p10 + b1 * a0; p11 = p11 + b1 * a1;
            let o2 = o + 32u;
            let c0 = wa[o2 + lid.x];
            let c1 = wa[o2 + lid.x + 16u];
            let e0 = wb[o2 + lid.y];
            let e1 = wb[o2 + lid.y + 16u];
            q00 = q00 + e0 * c0; q01 = q01 + e0 * c1;
            q10 = q10 + e1 * c0; q11 = q11 + e1 * c1;
        }
        s00 = s00 + (p00 + q00);
        s01 = s01 + (p01 + q01);
        s10 = s10 + (p10 + q10);
        s11 = s11 + (p11 + q11);
        workgroupBarrier();
    }
    let c0 = cbase + lid.x;
    let c1 = c0 + 16u;
    let r0 = rbase + lid.y;
    let r1 = r0 + 16u;
    if (r0 < cfg.r_rows && cfg.r_off + r0 < d) {
        if (c0 < d) { let ix = r0 * d + c0; h[ix] = h[ix] + s00; }
        if (c1 < d) { let ix = r0 * d + c1; h[ix] = h[ix] + s01; }
    }
    if (r1 < cfg.r_rows && cfg.r_off + r1 < d) {
        if (c0 < d) { let ix = r1 * d + c0; h[ix] = h[ix] + s10; }
        if (c1 < d) { let ix = r1 * d + c1; h[ix] = h[ix] + s11; }
    }
}
